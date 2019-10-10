# -*- coding: utf-8 -*-
"""
Analyzer for sinusoidal calibrations of seismometers.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
from glob import glob
from contextlib import redirect_stdout
from pkg_resources import get_distribution

from obspy import read, read_inventory
from obspy.core.inventory.response import \
    PolesZerosResponseStage, CoefficientsTypeResponseStage

from calan.core import Stft, get_clients, PACKAGE
from catalogue_tools.utilities import (
    MyArgumentParser, MyFormatter, get_logger)

MASS_KG = 5
GENERATOR_VMS = 629
COIL_OHM = 3600
CAL_OHM = 23
MOTOR_MS2A = GENERATOR_VMS/MASS_KG*CAL_OHM/COIL_OHM
MOTOR_MS2V = MOTOR_MS2A*CAL_OHM
DAC_GAIN = 2**12/5

LEN_FFT = 200
LEN_OVERLAP = int(LEN_FFT/2)
WINDOW = 'hann'
TRIM_S = 5
MIN_COHERENCE = 0.9
PATTERN = '*.mseed'
OUTPUT_LABEL = '_Output'
INPUT_LABEL = '_Input'
DPI = 150

THIS_FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(THIS_FILE_NAME)[0] + '.log'


def first_zero_crossing(trace, tol=0.01, time_type='matplotlib'):
    '''
    Estimate time of first zero crossing after first departure from zero
    greater than given tolerance relative to peak-to-peak amplitude. Only a
    very crude DC removal is attempted, by subtracting the value of the first
    sample.
    '''
    x = trace.data - trace.data[0]
    t = trace.times(type=time_type)

    i = ((2*np.abs(x)/(x.max() - x.min())) > tol).argmax()
    i += np.diff(np.sign(x[i:])).argmax()

    return t[i] - (t[i + 1] - t[i])/(x[i + 1] - x[i])*x[i]


class SynchronousCalibrationAnalyzer():

    def __init__(self, log_file_name=LOG_FILE_NAME, plot=False, dpi=DPI):

        # helpers
        self.logger = get_logger(__name__, log_file_name)
        self.plot = plot
        self.dpi = dpi
        self.client = get_clients()[0]

        # inputs
        self.output_file = None
        self.input_file = None
        self.test_name = None
        self.stream = None

        # intermediate results
        self.f_lim = None
        self.stft = None

        # outputs
        self.peak_variance = None
        self.gain_coherent = None
        self.phase_coherent = None

    def load_waveforms(self, output_file, trim_s=TRIM_S,
                       output_label=OUTPUT_LABEL, input_label=INPUT_LABEL):
        '''
        Load and preserve synchronous portions of input and output traces.
        '''
        input_file = output_file.replace(output_label, input_label)
        self.test_name = (os.path.splitext(output_file)[0]
                          .replace(output_label, ''))

        self.logger.info(output_file)
        output_stream = read(output_file)

        self.logger.info(input_file)
        input_stream = read(input_file)

        self.stream = input_stream + output_stream

        start = max(trace.stats.starttime for trace in self.stream) + trim_s
        end = min(trace.stats.endtime for trace in self.stream) - trim_s

        if self.plot:
            start_png = 'start_' + self.test_name + '.png'
            self.logger.info('Check start: ' + start_png)
            fig = self.stream.plot(endtime=start + 1,
                                   handle=True, equal_scale=False)
            for ax, trace in zip(fig.axes, reversed(self.stream)):
                ax.axvline(first_zero_crossing(trace), label='first zero',
                           linestyle='-.', color='red', linewidth=0.5)
                ax.axvline(start.matplotlib_date, label='start',
                           linestyle='--', color='blue', linewidth=0.5)
            fig.axes[0].legend(loc='upper right')
            fig.savefig(start_png, dpi=self.dpi)

            end_png = 'end_' + self.test_name + '.png'
            self.logger.info('Check end: ' + end_png)
            fig = self.stream.plot(starttime=end - 1,
                                   handle=True, equal_scale=False)
            for ax in fig.axes:
                ax.axvline(end.matplotlib_date, label='end',
                           linestyle='--', color='blue', linewidth=0.5)
            fig.axes[0].legend(loc='upper right')
            fig.savefig(end_png, dpi=self.dpi)

        self.stream = self.stream.trim(starttime=start, endtime=end)

        if self.stream.get_gaps():
            raise RuntimeError('Gaps detected.')

    def load_response(self, template_station='YKAR1', template_channel='SHZ',
                      motor_ms2v=MOTOR_MS2V, dac_gain=DAC_GAIN):
        '''
        Load nominal input and output responses.
        '''
        inventory_xml = self.stream[1].id + '.xml'
        # copy response from known station
        if not os.path.isfile(inventory_xml):
            inventory = self.client.get_stations(station=template_station,
                                                 channel=template_channel,
                                                 level='response')
            network = inventory[0]
            station = network[0]
            assert len(inventory) == len(network) == len(station) == 1
            network.code = self.stream[0].stats.network
            station.code = self.stream[0].stats.station
            station[0].location_code = self.stream[0].stats.location
            station[0].code = self.stream[0].stats.channel
            station.channels.append(station[0].copy())
            station[1].code = self.stream[1].stats.channel

            calibration = station[0].response
            fs = calibration.response_stages[2].decimation_input_sample_rate
            calibration.response_stages[0] = PolesZerosResponseStage(
                1, motor_ms2v, 0, 'M/S**2', 'V', 'LAPLACE (RADIANS/SECOND)', 0,
                [], [])
            calibration.response_stages[1] = CoefficientsTypeResponseStage(
                2, dac_gain, 0, 'V', 'COUNTS', 'DIGITAL',
                numerator=[], denominator=[], decimation_input_sample_rate=fs,
                decimation_factor=1, decimation_offset=0,
                decimation_delay=0, decimation_correction=0)
            calibration.recalculate_overall_sensitivity()

            self.logger.info('Writing: ' + inventory_xml)
            inventory.write(inventory_xml, 'STATIONXML')
        else:
            self.logger.info('Reading: ' + inventory_xml)
            inventory = read_inventory(inventory_xml)

        self.stream.attach_response(inventory)

    def compute_peak_response(self, len_fft=LEN_FFT, len_overlap=LEN_OVERLAP,
                              min_coherence=MIN_COHERENCE, window=WINDOW):
        '''
        Compute relative transfer function estimate at spectral peak.
        '''
        self.stft = Stft()
        self.stft.compute(self.stream[0].data, self.stream[1].data,
                          self.stream[0].stats.sampling_rate,
                          len_fft, len_overlap, window=window)
        self.f_lim = (self.stft.f[1], self.stft.f[-1])
        self.stft.trim()
        f = self.stft.f
        p_xx = np.mean(self.stft.p_xx, axis=1)
        p_yy = np.mean(self.stft.p_yy, axis=1)
        # H(f) = Y(f)/X(f) = voltage/acceleration = (voltage/velocity)/omega
        tfe = self.stft.get_transfer_function(alpha=0)
        gain = np.abs(tfe)
        phase = np.angle(tfe, deg=True)
        coherence = np.sqrt(self.stft.get_coherence_squared())
        variance = (1/coherence**2 - 1)/(2*self.stft.p_xx.shape[1])

        peak = p_xx.argmax()
        coherent = range(
            peak - (coherence[peak::-1] > min_coherence).argmin() + 1,
            peak + (coherence[peak:] > min_coherence).argmin())

        self.peak_variance = variance[peak]
        self.gain_coherent = (
            (gain[coherent]/variance[coherent]).sum() /
            (1/variance[coherent]).sum())
        self.phase_coherent = (
            (phase[coherent]/variance[coherent]).sum() /
            (1/variance[coherent]).sum())

        if self.plot:

            fig, axes = plt.subplots(4, 1, figsize=(6.5, 8), sharex=True)
            fig.subplots_adjust(hspace=0)

            axes[0].set_ylabel('Power [dB wrt cts$^2$/Hz]')
            axes[0].semilogx(f, 10*np.log10(p_xx), label='input')
            axes[0].semilogx(f, 10*np.log10(p_yy), label='output')
            axes[0].annotate(
                '%g Hz' % f[peak], (f[peak], 1.02),
                xycoords=axes[0].get_xaxis_transform(),
                annotation_clip=False, va='bottom', ha='center')

            axes[1].set_ylabel('Gain [dB]')
            axes[1].set_ylim((-40, 40))
            axes[1].semilogx(f, 20*np.log10(gain), color='black')
            gain_coherent_db = 20*np.log10(self.gain_coherent)
            axes[1].axhline(
                gain_coherent_db, linestyle='--', color='black', linewidth=0.5)
            axes[1].annotate(
                '%.2f dB' % gain_coherent_db, (1.02, gain_coherent_db),
                xycoords=axes[1].get_yaxis_transform(),
                annotation_clip=False, va='center', ha='left')

            axes[2].set_ylabel('Phase [°]')
            axes[2].set_yticks(range(-90, 91, 90))
            axes[2].set_ylim((-180, 180))
            axes[2].semilogx(f, phase, color='black')
            axes[2].axhline(
                self.phase_coherent,
                linestyle='--', color='black', linewidth=0.5)
            axes[2].annotate(
                '%.1f°' % self.phase_coherent, (1.02, self.phase_coherent),
                xycoords=axes[2].get_yaxis_transform(),
                annotation_clip=False, va='center', ha='left')

            axes[3].set_ylabel(r'Coherence, $\gamma$')
            axes[3].set_ylim((0, 1))
            axes[3].semilogx(f, coherence, color='black')
            axes[3].set_xlabel('Frequecy [Hz]')

            for i, ax in enumerate(axes):
                ax.axvline(f[peak],
                           linestyle='--', color='black', linewidth=0.5)
                ax.axvspan(f[coherent[0]], f[coherent[-1]],
                           color='0.3', alpha=0.3)

            axes[0].legend()
            fig.subplots_adjust()

            summary_png = ('spectra_%g-%gHz_%s.png' %
                           tuple(list(self.f_lim) + [self.test_name]))
            self.logger.info('Saving: ' + summary_png)
            fig.savefig(summary_png, dpi=self.dpi, bbox_inches='tight')

    def summary(self):
        result = pd.Series()
        result['test name'] = self.test_name
        result['channel id'] = self.stream[1].id
        result['start'] = self.stream[1].stats.starttime
        result['duration [s]'] = (self.stream[1].stats.endtime -
                                  self.stream[1].stats.starttime)
        result['f_min [Hz]'] = self.f_lim[0]
        result['f_max [Hz]'] = self.f_lim[1]
        result['windows'] = self.stft.p_xx.shape[1]
        result['gain'] = self.gain_coherent
        result['phase [°]'] = self.phase_coherent
        result['normalized error'] = np.sqrt(self.peak_variance)
        return result


def _argparser():
    '''
    Command-line arguments for main
    '''
    # pylint: disable=no-member
    parser = MyArgumentParser(prog=os.path.splitext(THIS_FILE_NAME)[0],
                              description=__doc__,
                              formatter_class=MyFormatter)

    parser.add_argument(
        '-g', '--pattern', default=PATTERN,
        help='glob pattern matching calibration files')
    parser.add_argument(
        '-l', '--len_fft', default=LEN_FFT, type=int,
        help='length of FFT')
    parser.add_argument(
        '-t', '--trim_s', default=TRIM_S, type=float,
        help='time to trim off beginning and end of signal, in seconds')
    parser.add_argument(
        '-w', '--window', default=WINDOW,
        help="window function to be used for Welch's method")
    parser.add_argument(
        '-i', '--input_label', default=INPUT_LABEL,
        help='string to be found in input waveform file names')
    parser.add_argument(
        '-o', '--output_label', default=OUTPUT_LABEL,
        help='string to be found in output waveform file names')
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='generate diagnostic plots for each analysis')
    parser.add_argument(
        '-d', '--dpi', default=DPI, type=int,
        help='resolution to use for plots in dots per inch')
    parser.add_argument(
        '-v', '--version', action='version',
        version='%s %s' % (PACKAGE, get_distribution(PACKAGE).version))
    return parser


def sine_analzyer(pattern=PATTERN, len_fft=LEN_FFT, window=WINDOW,
                  trim_s=TRIM_S,
                  output_label=OUTPUT_LABEL, input_label=INPUT_LABEL,
                  plot=False, dpi=DPI):
    '''
    Run analysis for all calibration files matching a glob pattern.
    '''
    analyzer = SynchronousCalibrationAnalyzer(plot=plot, dpi=dpi)

    output_csv = os.path.splitext(THIS_FILE_NAME)[0] + '.csv'
    if os.path.exists(output_csv) and os.path.isfile(output_csv) and \
            not os.access(output_csv, os.W_OK):
        analyzer.logger.error('Cannot write to ' + output_csv)
        return ''

    calibration_files = sorted([item for item in glob(pattern)
                                if output_label in item])
    if not calibration_files:
        analyzer.logger.error('No files matching "%s" contain "%s".' %
                              (pattern, output_label))
        return ''

    rows = []
    for calibration_file in calibration_files:
        try:
            analyzer.load_waveforms(calibration_file, input_label=input_label,
                                    output_label=output_label, trim_s=trim_s)
            analyzer.compute_peak_response(len_fft=len_fft, window=window)
            row = analyzer.summary()
            rows.append(row)
        except ValueError as ex:
            analyzer.logger.error(repr(ex))
            with StringIO() as buf, redirect_stdout(buf):
                analyzer.stream.print_gaps()
                gap_summary = buf.getvalue()
            analyzer.logger.debug('\n' + gap_summary)
        finally:
            plt.close('all')

    if not rows:
        analyzer.logger.error('No valid calibration results.')
        return ''

    df = pd.concat(rows, axis=1).T

    analyzer.logger.info('Summary: ' + output_csv)
    df.to_csv(output_csv, index=False)
    return output_csv


def main(argv=None):
    '''
    Return zero for successful termination, one otherwise.
    '''
    if argv is None:
        argv = sys.argv
    parser = _argparser()
    args = parser.parse_args(argv[1:])

    config = vars(args).copy()

    result = sine_analzyer(**config)

    return len(result) == 0


if __name__ == '__main__':
    sys.exit(main())
