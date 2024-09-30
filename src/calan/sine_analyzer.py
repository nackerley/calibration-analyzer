#!/usr/bin/env python3
"""
Analyzer for sinusoidal calibrations of seismometers.

Temperature data can be appended with provision of a weather station ID.
Use this inventory to look up the nearest wether station:
ftp://client_climate@ftp.tor.ec.gc.ca/Pub/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv

Limitations
-----------
Currently only supports calibration of one channel at at time.

Author: Nick Ackerley
"""
# pylint: disable=consider-using-f-string
import os
import sys
from pathlib import Path
from logging import getLogger
from io import StringIO
from glob import glob
from contextlib import redirect_stdout
from urllib.error import HTTPError
from urllib.parse import urlencode, urlunsplit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from timezonefinder import TimezoneFinder  # https://www.iana.org/time-zones

from obspy import read
from obspy.clients.fdsn import Client

from calan import PACKAGE_VERSION
from calan.core import start_logger, CHIS_PUBLIC_FDSNWS
from calan.utilities import MyArgumentParser, MyFormatter
from calan.stft import Stft

# default values for analysis
LEN_FFT = 200
LEN_OVERLAP = int(LEN_FFT/2)
WINDOW = 'hann'
TRIM_S = 5
MIN_COHERENCE = 0.9

# default inputs
PATTERN = '*.mseed'
OUTPUT_LABEL = '_Output'
INPUT_LABEL = '_Input'
DPI = 150

# weather station queries
WEATHER_SCHEME = 'https'
WEATHER_NETLOC = 'climate.weather.gc.ca'
WEATHER_PATH = '/climate_data/bulk_data_e.html'
WEATHER_STATION_ID = 51058
WEATHER_STATION_TIME_ZONE = 'America/Yellowknife'
WEATHER_QUERY = dict(format='csv', submit='Download+Data', timeframe=1)
WEATHER_TIME = 'Date/Time (LST)'

# bookkeeping
LOG_FILE_NAME = Path(__file__).stem + '.log'


def first_zero_crossing(trace, tol=0.01, time_type='matplotlib'):
    """
    Estimate time of first zero crossing.

    This is the time after first departure from zero
    greater than given tolerance relative to peak-to-peak amplitude. Only a
    very crude DC removal is attempted, by subtracting the value of the first
    sample.
    """
    x = trace.data - trace.data[0]
    t = trace.times(type=time_type)

    i = ((2*np.abs(x)/(x.max() - x.min())) > tol).argmax()
    i += np.diff(np.sign(x[i:])).argmax()

    return t[i] - (t[i + 1] - t[i])/(x[i + 1] - x[i])*x[i]


class SynchronousCalibrationAnalyzer():
    """Calibration analyzer for ObsPy stream including calibration signal."""

    TIME_ZONE_FINDER = TimezoneFinder()

    def __init__(self, plot=False, dpi=DPI):
        """Construct analyzer."""
        # helpers
        self.logger = getLogger(self.__class__.__name__)
        self.plot = plot
        self.dpi = dpi
        self.client = Client(CHIS_PUBLIC_FDSNWS)

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
        """Load and preserve synchronous portions of input & output traces."""
        input_file = output_file.replace(output_label, input_label)
        self.test_name = Path(output_file).stem.replace(output_label, '')

        self.logger.info('Output: %s', output_file)
        output_stream = read(output_file)

        self.logger.info('Input: %s', input_file)
        input_stream = read(input_file)

        self.stream = input_stream + output_stream

        start = max(trace.stats.starttime for trace in self.stream) + trim_s
        end = min(trace.stats.endtime for trace in self.stream) - trim_s

        if self.plot:
            start_png = Path(self.test_name).parent.joinpath(
                'start_%s.png' % Path(self.test_name).name)
            self.logger.info('Check start: %s', start_png)
            fig = self.stream.plot(endtime=start + 1,
                                   handle=True, equal_scale=False)
            for ax, trace in zip(fig.axes, reversed(self.stream)):
                ax.axvline(first_zero_crossing(trace), label='first zero',
                           linestyle='-.', color='red', linewidth=0.5)
                ax.axvline(start.matplotlib_date, label='start',
                           linestyle='--', color='blue', linewidth=0.5)
            fig.axes[0].legend(loc='upper right')
            fig.savefig(start_png, dpi=self.dpi)

            end_png = Path(self.test_name).parent.joinpath(
                'end_%s.png' % Path(self.test_name).name)
            self.logger.info('Check end: %s', end_png)
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

    def compute_peak_response(self, len_fft=LEN_FFT, len_overlap=LEN_OVERLAP,
                              min_rel_power=1e-4, window=WINDOW):
        """Compute relative transfer function estimate at spectral peak."""
        self.stft = Stft()
        self.stft.compute(self.stream[0].data, self.stream[1].data,
                          self.stream[0].stats.sampling_rate,
                          len_fft, len_overlap, window=window)
        self.f_lim = (self.stft.f[1], self.stft.f[-1])
        self.stft.trim()
        f = self.stft.f
        p_xx = self.stft.mean(self.stft.p_xx).squeeze()
        p_yy = self.stft.mean(self.stft.p_yy).squeeze()
        tfe = self.stft.tf_estimate(alpha=0).squeeze()
        gain = np.abs(tfe)
        phase = np.angle(tfe, deg=True)
        coherence = np.sqrt(self.stft.coherence_squared()).squeeze()
        variance = self.stft.variance().squeeze()

        peak = p_xx.argmax()
        coherent = range(
            peak - (p_xx[peak::-1]/p_xx[peak] > min_rel_power).argmin() + 1,
            peak + (p_xx[peak:]/p_xx[peak] > min_rel_power).argmin())

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

            for ax in axes:
                ax.axvline(f[peak],
                           linestyle='--', color='black', linewidth=0.5)
                ax.axvspan(f[coherent[0]], f[coherent[-1]],
                           color='0.3', alpha=0.3)

            axes[0].legend()
            fig.subplots_adjust()

            summary_png = Path(self.test_name).parent.joinpath(
                'spectra_%g-%gHz_%s.png' % tuple(
                    list(self.f_lim) + [Path(self.test_name).name]))

            getLogger(__name__).info('Saving: %s', summary_png)
            fig.savefig(summary_png, dpi=self.dpi, bbox_inches='tight')

    def get_temperature(self, dt_utc, station_id=WEATHER_STATION_ID,
                        time_zone=WEATHER_STATION_TIME_ZONE):
        """Get temperature near station at given time."""
        time_zone = pytz.timezone(time_zone)
        dt_local = time_zone.fromutc(dt_utc)

        query = dict(stationID=station_id, Year=dt_local.year,
                     Month=dt_local.month, Day=dt_local.day)
        query.update(WEATHER_QUERY)
        url = urlunsplit((WEATHER_SCHEME, WEATHER_NETLOC, WEATHER_PATH,
                          urlencode(query), ''))
        try:
            df = pd.read_csv(url, parse_dates=[WEATHER_TIME])
            actual_tz = pytz.timezone(self.TIME_ZONE_FINDER.timezone_at(
                lat=df['Latitude (y)'].mean(), lng=df['Longitude (x)'].mean()))
            if actual_tz != time_zone:
                self.logger.warning(
                    'Data is from time zone "%s"; expected "%s".',
                    actual_tz.zone, time_zone)

            df[WEATHER_TIME] = df[WEATHER_TIME].dt.tz_localize(
                time_zone, ambiguous=True, nonexistent='NaT')
            df[WEATHER_TIME] = df[WEATHER_TIME].dt.tz_convert(None)

            index = (df[WEATHER_TIME] > dt_utc).idxmax()
            result = df.at[index, 'Temp (°C)']
        except HTTPError as ex:
            self.logger.error(repr(ex))
            result = np.nan

        self.logger.info('Temperature: %g°C', result)
        return result

    def summary(self, station_id=WEATHER_STATION_ID,
                time_zone=WEATHER_STATION_TIME_ZONE):
        """Summarize result."""
        result = pd.Series(dtype=object)
        result['test name'] = self.test_name
        result['channel id'] = self.stream[1].id
        result['start'] = pd.to_datetime(
            self.stream[1].stats.starttime.datetime)
        result['duration [s]'] = (self.stream[1].stats.endtime -
                                  self.stream[1].stats.starttime)
        result['f_min [Hz]'] = self.f_lim[0]
        result['f_max [Hz]'] = self.f_lim[1]
        result['windows'] = self.stft.p_xx.shape[1]
        result['f_peak [Hz]'] = self.stft.f[
            np.mean(self.stft.p_xx, axis=2).squeeze().argmax()]
        result['gain [dB]'] = 20*np.log10(self.gain_coherent)
        result['phase [°]'] = self.phase_coherent
        result['normalized error'] = np.sqrt(self.peak_variance)
        result['temperature [°C]'] = self.get_temperature(
            self.stream[1].stats.starttime.datetime, station_id, time_zone)

        return result


def _argparser():
    """Command-line interface."""
    # pylint: disable=no-member
    parser = MyArgumentParser(prog=Path(__file__).stem,
                              description=__doc__,
                              formatter_class=MyFormatter)

    parser.add_argument(
        '-g', '--pattern', default=PATTERN,
        help='glob pattern matching calibration files')
    parser.add_argument(
        '-l', '--len-fft', default=LEN_FFT, type=int,
        help='length of FFT')
    parser.add_argument(
        '-t', '--trim-s', default=TRIM_S, type=float,
        help='time to trim off beginning and end of signal, in seconds')
    parser.add_argument(
        '-w', '--window', default=WINDOW,
        help="window function to be used for Welch's method")
    parser.add_argument(
        '-i', '--input-label', default=INPUT_LABEL,
        help='string to be found in input waveform file names')
    parser.add_argument(
        '-o', '--output-label', default=OUTPUT_LABEL,
        help='string to be found in output waveform file names')
    parser.add_argument(
        '--summary-csv', default='',
        help='by default a file name is generated from --pattern')
    parser.add_argument(
        '-s', '--station-id', default=WEATHER_STATION_ID,
        help='weather station id for temperature lookup')
    parser.add_argument(
        '-z', '--time-zone', default=WEATHER_STATION_TIME_ZONE,
        help='time zone of weather station')
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='generate diagnostic plots for each analysis')
    parser.add_argument(
        '-d', '--dpi', default=DPI, type=int,
        help='resolution to use for plots in dots per inch')
    parser.add_argument(
        '-v', '--version', action='version',
        version=PACKAGE_VERSION)
    return parser


def sine_analzyer(pattern=PATTERN, len_fft=LEN_FFT, window=WINDOW,
                  trim_s=TRIM_S, station_id=WEATHER_STATION_ID,
                  time_zone=WEATHER_STATION_TIME_ZONE,
                  output_label=OUTPUT_LABEL, input_label=INPUT_LABEL,
                  summary_csv='',
                  plot=False, dpi=DPI):
    """Run analysis for all calibration files matching a glob pattern."""
    analyzer = SynchronousCalibrationAnalyzer(plot=plot, dpi=dpi)

    if not summary_csv:
        pattern_slug = ''.join(char for char in Path(pattern).stem
                               if char.isalnum())
        output_parts = [Path(__file__).stem]
        if pattern_slug:
            output_parts += pattern_slug.split('_')
        summary_csv = '_'.join(output_parts) + '.csv'

    if Path(summary_csv).exists() and Path(summary_csv).is_file() and \
            not os.access(summary_csv, os.W_OK):
        analyzer.logger.error(
            'Will not be able to write summary to %s.', summary_csv)
        return ''

    calibration_files = sorted([item for item in glob(pattern)
                                if output_label in item])
    if not calibration_files:
        analyzer.logger.error(
            'No files matching pattern "%s" contain output label "%s".',
            pattern, output_label)
        return ''

    rows = []
    for calibration_file in calibration_files:
        try:
            analyzer.load_waveforms(calibration_file, input_label=input_label,
                                    output_label=output_label, trim_s=trim_s)
            analyzer.compute_peak_response(len_fft=len_fft, window=window)
            row = analyzer.summary(station_id=station_id, time_zone=time_zone)
            rows.append(row)
        except ValueError as ex:
            analyzer.logger.error(repr(ex))
            with StringIO() as buf, redirect_stdout(buf):
                analyzer.stream.print_gaps()
                gap_summary = buf.getvalue()
            analyzer.logger.debug('\n%s', gap_summary)
        finally:
            plt.close('all')

    if not rows:
        analyzer.logger.error('No valid calibration results.')
        return ''

    df = pd.concat(rows, axis=1).T
    df['start'] = pd.to_datetime(df['start'])
    df.sort_values('start', inplace=True)

    # clean up column data types - not strictly necessary
    df['duration [s]'] = df['duration [s]'].astype(float)
    df['f_min [Hz]'] = df['f_min [Hz]'].astype(float)
    df['f_max [Hz]'] = df['f_max [Hz]'].astype(float)
    df['windows'] = df['windows'].astype(int)
    df['f_peak [Hz]'] = df['f_peak [Hz]'].astype(float)
    df['gain [dB]'] = df['gain [dB]'].astype(float).round(4)
    df['phase [°]'] = df['phase [°]'].astype(float).round(3)
    df['normalized error'] = [float('%.1e' % item)
                              for item in df['normalized error']]
    df['temperature [°C]'] = df['temperature [°C]'].astype(float)

    analyzer.logger.info('Summary: %s', summary_csv)
    df.to_csv(summary_csv, index=False)
    return summary_csv


def main(argv=None):
    """Analyze calibrations and return system exit code."""
    if argv is None:
        argv = sys.argv
    parser = _argparser()
    args = parser.parse_args(argv[1:])

    if Path(LOG_FILE_NAME).is_file():
        Path(LOG_FILE_NAME).unlink()
    start_logger(__name__, LOG_FILE_NAME, 'INFO')

    config = vars(args).copy()

    result = sine_analzyer(**config)

    return len(result) == 0


if __name__ == '__main__':
    sys.exit(main())
