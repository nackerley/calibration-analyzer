# -*- coding: utf-8 -*-
"""
Analyzer for broadband calibrations of seismometers using arbitrary signals.

LIMITATIONS:
  * calibration input poles and zeros are hardcoded for Trillium 120QA

Author: Nick Ackerley
"""
import os
import sys
import lzma
import numpy as np
from glob import glob
from pkg_resources import get_distribution
from operator import mul
from functools import reduce
from math import log10, floor, ceil
from collections import OrderedDict
from copy import deepcopy
from tempfile import gettempdir
from scipy import signal
from scipy.special import erfinv
import matplotlib.pyplot as plt
from string import Template

import pandas as pd
import statsmodels.api as sm

from obspy import read, read_inventory, Trace, Stream, UTCDateTime
from obspy.signal.invsim import simulate_seismometer

from catalogue_tools.core import PACKAGE, short_utc
from catalogue_tools.utilities import (
    pretty_duration, round_sig, logspace, get_logger,
    MyArgumentParser, MyFormatter)

from calan.core import (
    Stft, factor_names, subplots_squeeze, inventory_items,
    lti_from_zpsf, minreal, unwrap_mid,
    len_fft_welch, num_windows_welch,
    extract_decimation_coefficients, multi_decim)
from calan.calibration_toolbox import (
    sample_hold_digitize, pad_for_decimation)

# %% setup
pd.plotting.register_matplotlib_converters()

# %% defaults

# default values for analysis
DEFAULT_NUM_WINDOWS = 30
DEFAULT_WINDOW = 'hann'

# inputs
DEFAULT_OUTPUT_PATTERN = '*.mseed'
DEFAULT_RESPONSE_PATTERN = '*.resp'
DEFAULT_CALIBRATION_FILE = 'prb_1V_10ms_3h.lzma'
DEFAULT_DPI = 120

# Centaur configuration
DEFAULT_LEAD_IN = 360
DEFAULT_LEAD_OUT = 360
DEFAULT_PRE_TIME = 10
DEFAULT_POST_TIME = 10
DEFAULT_DELAY_START = 0

# %% constants
THIS_FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(THIS_FILE_NAME)[0] + '.log'

# Nanometrics Centaur User Guide 17935R5, 2016-11-02
FULL_SCALE_VOLTAGE = 5
DAC_BITS = 16
DAC_GAIN = FULL_SCALE_VOLTAGE/(2**(DAC_BITS - 1) - 1)
SAMPLE_FORMAT = '<h'  # little-endian short (16-bit) integer
CALIBRATION_DTYPE = 'int%d' % DAC_BITS

# Nanometrics Trillium 120Q/QA User Guide 17751R2, 2015-10-05
CAL_ZEROS = ()  # rad/s
CAL_POLES = (-160, -3177)  # rad/s
CAL_SENS = -0.01023  # m/s^2/V (Table 12.1, not 11.2)
CAL_FREQ = 0.001  # Hz

# IDC
TEST_BAND_HZ = (0.02, 16)
MAX_AMPLITUDE_PERCENT = 5
MAX_PHASE_DEGREES = 5

CAL_RESULT_MSG_ID = 'CAL_{year:4d}_1_{station}_CR'
IMS_DATETIME_FMT = '%Y/%m/%d %H:%M'

IMS_HEADER = Template('''\
begin IMS2.0
msg_type COMMAND_RESPONSE
msg_id $msg_id
ref_id $ref_id
time_stamp $time_stamp
''')
RESPONSE_HEADER = Template('''
sta_list $station
chan_list $channel
CALIBRATE_RESULT
in_spec $in_spec
data_type RESPONSE IMS2.0
''')
CAL_BLOCK = (
    'CAL2 {station:5.5s} {channel:3.3s} {aux_id:4.4s} {inst_type:6.6s} '
    '{calib:15.8e} {calper:7.3f} {sample_rate:11.5f} {start} {end}\n')
FAP_HEADER = (
    'FAP2 {stage:2d} {units:1.1s} {decimation:4.4s} {group_correction:8.3f} '
    '{count:3d} {description:25.25s}\n')
FAP_DATA = ' {frequency:10.5f} {amplitude:15.8e} {phase:4.0f}\n'
MAX_FAP_LEN = 999
IMS_FOOTER = 'stop\n'


# %% definitions
def _argparser():
    '''
    Command-line arguments for main
    '''
    # pylint: disable=no-member
    parser = MyArgumentParser(prog=os.path.splitext(THIS_FILE_NAME)[0],
                              description=__doc__,
                              formatter_class=MyFormatter)

    parser.add_argument(
        '-g', '--pattern', default=DEFAULT_OUTPUT_PATTERN,
        help='glob pattern matching calibration output files, '
        'in any format readable by obspy.read()')
    parser.add_argument(
        '-n', '--num_windows', default=DEFAULT_NUM_WINDOWS, type=int,
        help="number of windows to used for Welch's method")
    parser.add_argument(
        '--len_fft', default=None, type=int,
        help="window length for Welch's method, overrides num_windows")
    parser.add_argument(
        '-w', '--window', default=DEFAULT_WINDOW,
        help="window function to be used for Welch's method")
    parser.add_argument(
        '-r', '--response_pattern', default=DEFAULT_RESPONSE_PATTERN,
        help='calibration response file, '
        'in any format readable by obspy.read_inventory()')
    parser.add_argument(
        '-c', '--calibration_file', default=DEFAULT_CALIBRATION_FILE,
        help='calibration input file, lzma compressed')
    parser.add_argument(
        '-d', '--delay_start', default=DEFAULT_DELAY_START, type=float,
        help='amount to delay calibration start time, in seconds')
    parser.add_argument(
        '-b', '--test_band_hz', nargs=2, type=float, default=TEST_BAND_HZ,
        metavar=('MIN_FREQUENCY_HZ', 'MAX_FREQUENCY_HZ'),
        help='frequency band, in Hz, over which to apply test limits')
    parser.add_argument(
        '-t', '--test_limits', nargs=2, type=float,
        default=(MAX_AMPLITUDE_PERCENT, MAX_PHASE_DEGREES),
        metavar=('MAX_AMPLITUDE_PERCENT', 'MAX_PHASE_DEGREES'),
        help='maximum deviation from nominal of amplitude in percent and '
        'phase in degrees')
    parser.add_argument(
        '--write_ims', action='store_true',
        help='write IMS2.0 CALIBRATE_RESULT message with FAP2 payload')
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='generate basic plots (start-check, transfer function and '
        'variance) for each analysis')
    parser.add_argument(
        '--diagnostic', action='store_true',
        help='generate extra diagnostic plots for each analysis')
    parser.add_argument(
        '--dpi', default=DEFAULT_DPI, type=int,
        help='resolution to use for plots in dots per inch')
    parser.add_argument(
        '-v', '--version', action='version',
        version='%s %s' % (PACKAGE, get_distribution(PACKAGE).version))
    return parser


def seismometer_calibration_lti(zeros=CAL_ZEROS, poles=CAL_POLES,
                                sensitivity=CAL_SENS, frequency=CAL_FREQ):
    '''
    Returns a seismometer calibration input transfer function
    :class:`scipy.signal.lti`. Default values are for Trillium 120Q.
    Sensitivity is provided as [m/s^2/V] but lti returned is [(m/s)/V].
    '''

    cal_lti = lti_from_zpsf(zeros, poles, sensitivity, frequency)
    cal_lti.poles = np.concatenate((cal_lti.poles, np.zeros(1)))
    # cal_lti.gain = 2*np.pi*cal_lti.gain
    cal_lti = minreal(cal_lti)
    cal_lti.units = '(m/s)/V'
    return cal_lti


def accelerometer_calibration_lti(zeros=(), poles=(),
                                  sensitivity=1, frequency=1):
    '''
    Returns a accelerometer calibration input transfer function
    :class:`scipy.signal.lti`. Default values are for a generic accelerometer
    calibration circuit at 1 m/s^2/V.
    '''

    cal_lti = lti_from_zpsf(zeros, poles, sensitivity, frequency)
    cal_lti = minreal(cal_lti)
    cal_lti.units = '(m/s^2)/V'
    return cal_lti


def convert_volts_to_counts(signal):
    '''
    Convert a calibration signal from volts to counts.

    See also docstring for convert_counts_to_volts.
    '''
    signal = np.array(signal, dtype='float')/DAC_GAIN
    return signal.astype(np.dtype('int%d' % DAC_BITS))


def convert_counts_to_volts(signal):
    '''
    Convert a calibration signal from counts to volts.

    DAC gain is set according to statement "The maximum output signal (+5 V)
    corresponds to the maximum value (32 767)" (p. 97 of "Centaur User
    Guide 17935R5.pdf") and assumption that 0 counts gives 0 V. Thus the rest
    of table on p.97 of the user guide is in error.
    '''
    signal = np.array(signal, dtype=np.dtype('int%d' % DAC_BITS))
    return signal.astype('float')*DAC_GAIN


class Fit():
    '''
    Container for transfer function fits.
    '''

    def __init__(self):
        self.timing = None
        self.gain = None

    def __str__(self, confidence=0.95):
        lines = [self.__class__.__name__ + ':']
        lines.append('\t' + self.timing_summary(confidence=confidence))
        lines.append('\t' + self.gain_summary(confidence=confidence))
        return '\n'.join(lines)

    @staticmethod
    def _num_sigma(confidence=0.95):
        return np.sqrt(2)*erfinv(confidence)

    def timing_summary(self, confidence=0.95):
        if self.timing is None:
            return 'timing fit: None'
        uncertainty = self._num_sigma()*self.timing.bse
        digits = round(log10(abs(self.timing.params)/uncertainty)) + 1
        return (
            'timing error %s ±%s (%.0f%% conf.)' %
            (pretty_duration(self.timing.params, fmt=digits, thresh=0.05),
             pretty_duration(uncertainty, fmt=1, thresh=0.05),
             100*confidence))

    def gain_summary(self, confidence=0.95):
        if self.gain is None:
            return 'gain fit: None'
        uncertainty = self._num_sigma()*self.gain.bse
        digits = round(log10(abs(self.gain.params - 1)/uncertainty)) + 1
        return (
            'gain error %g%% ±%g%% (%.0f%% conf.)' %
            (round_sig(100*(self.gain.params - 1), digits),
             round_sig(100*self._num_sigma()*self.timing.bse, 1),
             100*confidence))


class CalibrationAnalyzer():
    '''
    A calibration signal analyzer based on :class:`obspy.Stream`.
    '''
    CACHE_FORMAT = 'MSEED'

    def __init__(self, log_level='INFO', savefig=True):
        '''
        Sets up data server and calibration details for later use.

        Arguments
        ---------
        str: log_level
            console logging level
        '''
        logger = get_logger(self.__class__.__name__, LOG_FILE_NAME, log_level)
        logger.info('%s: %s' % (PACKAGE, get_distribution(PACKAGE).version))

        self.info = OrderedDict((
            ('waveform_file', ''),
            ('response_file', []),
            ('calibration_file', ''),
            ('start', None),
            ('end', None),
            ('spec_min_freq_hz', np.NaN),
            ('spec_max_freq_hz', np.NaN),
            ('spec_max_amp_pct', np.NaN),
            ('spec_max_phase_deg', np.NaN),
            ('in_spec', []),
            ))
        self.stream = Stream()

        self.lti = OrderedDict((
            ('sensor', None),
            ('cal', None),
            ('system', None),
            ))
        self.stft = Stft()
        self.savefig = savefig

        self.fit = Fit()

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        lines += ['\tInfo:']
        for key, value in self.info.items():
            lines.append('\t\t%s: %s' % (key, value))
        lines += ['\t' + line
                  for line in self.stream.__str__().strip().split('\n')]
        lines += ['\tNominal:']
        for key, value in self.lti.items():
            lines.append('\t\t%s: %s' % (key, value))
        lines += ['\t' + line
                  for line in self.stft.__str__().strip().split('\n')]
        lines += ['\t' + line
                  for line in self.fit.__str__().strip().split('\n')]
        return '\n'.join(lines)

    def __repr__(self):
        return self.__str__()

    def load_stream(self, waveform_file=None, lead_in=DEFAULT_LEAD_IN,
                    lead_out=DEFAULT_LEAD_OUT, pre_time=DEFAULT_PRE_TIME,
                    post_time=DEFAULT_POST_TIME,
                    adjust_start=DEFAULT_DELAY_START):
        '''
        Load calibration waveforms.

        Two cases are supported:
            1.  The stream does not include a calibration signal:
                The calibration start and end are inferred from lead
                in/out and event pre/post times.
            2.  The stream includes a calibration signal:
                The calibration start and end are that of the stream.
        '''
        logger = get_logger(__name__)
        logger.info(waveform_file)
        self.stream = read(waveform_file).sort()
        self.info['waveform_file'] = waveform_file
        self.info['start'] = self.stream[0].stats.starttime
        self.info['end'] = self.stream[0].stats.endtime

        calibration_trace = next((trace for trace in self.stream
                                  if trace.id[-1] == 'C'), None)
        if calibration_trace:
            self.stream.traces = [  # move calibration trace to end
                trace for trace in self.stream
                if trace.id != calibration_trace.id] + [calibration_trace]
        else:
            self.info['start'] += lead_in + pre_time + adjust_start
            self.info['end'] -= lead_out + post_time - adjust_start

    def sampling_rate(self):
        '''Return stream sampling rate'''

        return self.stream[0].stats.sampling_rate

    def pre_seconds(self):
        '''Return sum of lead-in and event pre-time.'''

        return self.info['start'] - self.stream[0].stats.starttime

    def post_seconds(self):
        '''Return sum of lead-out and event post-time'''

        return self.stream[0].stats.endtime - self.info['end']

    def load_response(self, response_pattern, ignore_open_closed=True,
                      force_first=False):
        '''
        Load response file describing the system being calibrated.
        '''
        logger = get_logger(__name__)
        logger.info(response_pattern)
        response_files = glob(response_pattern)

        self.info['response_file'] = []
        inventory = None
        for trace in self.stream:
            for response_file in response_files:
                try:
                    trace.attach_response(read_inventory(response_file))
                    logger.info('Found %s: %s' % (trace.id, response_file))
                    self.info['response_file'].append(response_file)

                    if inventory is None:
                        inventory = read_inventory(response_file)
                    else:
                        inventory += read_inventory(response_file)
                    break
                except ValueError:
                    continue

        earliest = min(trace.stats.starttime for trace in self.stream)
        latest = max(trace.stats.endtime for trace in self.stream)

        if force_first:
            inventory.networks = [inventory[0]]
            inventory[0].stations = [inventory[0][0]]
            inventory[0][0].channels = [inventory[0][0][0]]
            for trace in self.stream:
                network_code, station_code, location_code, channel_code = \
                    trace.id.split('.')
                network = next((network for network in inventory
                                if network.code == network_code), None)
                if network is None:
                    network = deepcopy(inventory[0])
                    network.code = network_code
                    inventory.networks.append(network)
                station = next((station for station in network
                                if station.code == station_code), None)
                if station is None:
                    station = deepcopy(network[0])
                    station.code = station_code
                    network.stations.append(station)
                channel = next((channel for channel in station
                                if channel.code == channel_code and
                                channel.location_code == location_code), None)
                if channel is None:
                    channel = deepcopy(station[0])
                    channel.code = channel_code
                    channel.location_code = location_code
                    station.channels.append(channel)

        if ignore_open_closed:
            for _, _, channel in inventory_items(inventory):
                channel.starttime = earliest
                channel.endtime = latest

        not_found = self.stream.attach_response(inventory)
        if not_found:
            logger.warning(
                'No response found:' +
                ', '.join([trace.id for trace in not_found]))

    def _pad_lead_in_out(self, signal, factor):
        sampling_rate = self.sampling_rate()*factor
        return np.hstack((
            np.zeros(int(round(self.pre_seconds()*sampling_rate))),
            signal,
            np.zeros(int(round(self.post_seconds()*sampling_rate)))))

    def load_calibration(self, calibration_file):
        '''
        No attempt is made to match the response to the exact channel used;
        it is assumed that all traces in the calibation stream have the same
        nominal response.
        '''
        logger = get_logger(__name__)
        if any(trace.id[-1] == 'C' for trace in self.stream):
            logger.error('Waveform data already includes calibration channel')
            return

        logger.info(calibration_file)
        self.info['calibration_file'] = calibration_file

        b_stages, factors = extract_decimation_coefficients(
            self.stream[0].stats.response.response_stages)
        factor = reduce(mul, factors)
        phase_suffix = ''
        if factors[-1] == 2:
            if len(b_stages[-1]) == 223:
                phase_suffix = '_linear'
            elif len(b_stages[-1]) == 110:
                phase_suffix = '_minimum'
        if not phase_suffix:
            logger.warning('Cannot determine whether decimation filters are '
                           'minimum or linear phase.')

        cache_file = os.path.join(
            gettempdir(),
            '%s_%gsps%s.%s' % (
                calibration_file.replace('.lzma', ''),
                self.stream[0].stats.sampling_rate, phase_suffix,
                self.CACHE_FORMAT.lower()))
        if os.path.isfile(cache_file):
            logger.info('Found cache: %s' % cache_file)
            signal = read(cache_file)[0].data
        else:
            logger.info('Reading: %s' % calibration_file)
            signal = np.frombuffer(lzma.open(calibration_file).read(),
                                   dtype=CALIBRATION_DTYPE)
            signal = sample_hold_digitize(signal)
            signal = pad_for_decimation(signal, b_stages, factors)[0]
            signal = self._pad_lead_in_out(signal, factor)

            logger.info('Decimating by %d ...' % factor)
            signal = multi_decim(signal, b_stages, factors)[0]

        if len(signal) != self.stream[0].stats.npts:
            logger.error(
                'Expected %d, obtained %d points for calibration signal' %
                (self.stream[0].stats.npts, len(signal)))
            raise RuntimeError('Check lead-in and lead-out times.')

        stats = {key: self.stream[0].stats[key] for key in
                 ['network', 'station', 'location', 'sampling_rate']}
        stats['channel'] = self.stream[0].stats['channel'][:2] + 'C'
        stats['starttime'] = self.stream[0].stats.starttime
        trace = Trace(data=np.ascontiguousarray(signal), header=stats)
        stream = Stream([trace])

        if not os.path.isfile(cache_file):
            logger.info('Caching: %s' % cache_file)
            stream.write(cache_file, format=self.CACHE_FORMAT)
        self.stream += stream

    def _sensor_stage(self):
        '''
        Note assumption that sensor is a single stage.
        '''
        return self.stream[0].stats.response.response_stages[0]

    def setup_nominal_response(self, cal_lti=None):
        '''
        Set up nominal responses for the calibration input and sensor response.
        For seismometers response units are [m/s/V] and [V/(m/s)] respectively.
        For accelerometers response units are [m/s^2/V] and [V/(m/s^2)]
        respectively.

        The default value of 'None' results in the calibration response of the
        Trillium 120Q being used. For other seismometers the simplest way to
        specify the transfer function is to set cal_lti to a tuple of
        (zeros, poles, gain). For most accelerometers cal_lti can simply
        be set to the sensitivity of the calibration input circuit,
        i.e. no zeros, no poles.

        The sensor response is read from the dataless SEED provided to
        :func:`load_calibration` if given, otherwise from the response read
        by :func:`load_stream`.

        The system transfer function is the product of the calibration and
        sensor tranfer functions, and can be used, for example, to simulate
        the response of the sensor to a calibration.
        '''
        if cal_lti is None:
            self.lti['cal'] = seismometer_calibration_lti()
        else:
            self.lti['cal'] = signal.lti(cal_lti)

        stage = self._sensor_stage()
        self.lti['sensor'] = lti_from_zpsf(stage.zeros, stage.poles,
                                           stage.stage_gain,
                                           stage.normalization_frequency)

        # nominal response is calibration response * sensor response
        total_zeros = (self.lti['cal'].zeros.tolist() +
                       self.lti['sensor'].zeros.tolist())
        total_poles = (self.lti['cal'].poles.tolist() +
                       self.lti['sensor'].poles.tolist())
        total_gain = self.lti['cal'].gain*self.lti['sensor'].gain
        nominal_lti = signal.lti(total_zeros, total_poles, total_gain)
        self.lti['system'] = minreal(nominal_lti)

    def get_stream(self, which='output', trim=True):
        '''
        Retrieve input or output stream, with padding optionally trimmed.
        '''
        assert which in ['input', 'output', 'simulated']
        stream = self.stream.copy()

        if which in ['input', 'simulated']:
            stream.traces = [trace for trace in stream if trace.id[-1] == 'C']
        else:
            stream.traces = [trace for trace in stream if trace.id[-1] != 'C']

        if trim:
            stream = stream.trim(self.info['start'], self.info['end'])

        return stream

    def get_digitizer_sensitivity(self):
        '''
        Get sensitivity of digitizer for input stream.
        '''
        return float(
            self.stream[0].stats.response.instrument_sensitivity.value /
            self._sensor_stage().stage_gain)

    def output_voltage(self, trim=True):
        '''
        Convert output from stream [counts] to array [V].

        Arguments
        ---------
        trim: bool
            trim padding (e.g. turn-on and turn-off times)

        Returns
        -------
        signal: :class:`numpy.ndarray`
            output [V], one row per channel
        '''
        stream = self.get_stream('output', trim=trim)
        signal = np.row_stack([trace.data for trace in stream]).astype(float)
        signal /= self.get_digitizer_sensitivity()

        return signal

    def input_voltage(self, trim=True):
        '''
        Convert input from stream [counts] to array [V]

        Arguments
        ---------
        trim: bool
            trim padding (e.g. turn-on and turn-off times)

        Returns
        -------
        signal: :class:`numpy.array`
            input [V]
        '''
        stream = self.get_stream('input', trim=trim)
        signal = convert_counts_to_volts(stream[0].data)

        return signal

    def compute(self, len_fft=None, num_windows=30, fraction_overlap=0.5):
        '''
        Time-consuming part of calibration analysis is done here.

        Each segment is detrended by removing a constant value before
        application of a 'hanning' window.

        The length of FFT segments can be specified directly, and is efficient
        for any number with a large number of small factors (not just powers
        of 2), but it is generally more convenient to specify a target number
        of segments as it is the (actual) number of segments in Welch's method
        which reduces the variance in the result.
        '''
        x = self.input_voltage()
        y = self.output_voltage()
        f_sample = self.sampling_rate()

        if len_fft is None:
            len_fft = len_fft_welch(len(x), num_windows, fraction_overlap)
        len_overlap = int(fraction_overlap*len_fft)

        num_windows = num_windows_welch(len(x), len_fft, len_overlap)

        self.stft.compute(x, y, f_sample, len_fft, len_overlap)
        self.stft.trim()

    def summary(self, min_coherence=None):
        '''
        Summarize calibration result, one row per trace.

        Columns are pd.MultiIndex, with first level:
            'info', 'magnitude_db', 'phase_deg', 'variance_db'

        The second level of the pd.MultiIndex (except for info) is frequency.

        Rows are indexed by obspy.Trace.id and start time.

        Returns
        -------
        df : pandas DataFrame
            calibration summary table
        '''
        f = pd.Index(self.stft.f, name='f')
        tf_estimate = (self.stft.tf_estimate() /
                       signal.freqresp(self.lti['cal'], 2*np.pi*f)[1])
        tf_nominal = signal.freqresp(self.lti['sensor'], 2*np.pi*f)[1]

        index = pd.MultiIndex.from_tuples(
            [(pd.to_datetime(self.info['start'].datetime)
              if trace.id[-1] != 'C' else pd.NaT, trace.id)
             for trace in self.stream],
            names=['start', 'trace_id'])

        # first construct info for calibration outputs
        info = pd.DataFrame(index=index[:-1])

        for key, value in pd.Series(self.info).drop(
                ['start', 'end', 'calibration_file']).items():
            info[key] = value
        info['duration [s]'] = self.info['end'] - self.info['start']
        info['pre [s]'] = self.pre_seconds()
        info['post [s]'] = self.post_seconds()
        info['sampling rate [sps]'] = self.sampling_rate()
        info['windows'] = self.stft.p_xx.shape[1]
        info['windows'] = info['windows'].astype(int)

        # now append info for calibration input
        info = info.append(pd.Series(name=index[-1], dtype=float))
        info.loc[info.index[-1], 'waveform_file'] = \
            self.info['calibration_file']
        info['output units'] = self._sensor_stage().output_units
        info['input units'] = self._sensor_stage().input_units.lower()
        info[PACKAGE] = get_distribution(PACKAGE).version

        magnitude = pd.DataFrame(np.vstack(
            (20*np.log10(np.abs(tf_estimate)),
             20*np.log10(np.abs(tf_nominal)))),
            columns=f, index=index).round(3)
        phase = pd.DataFrame(np.vstack(
            (np.angle(tf_estimate, deg=True),
             np.angle(tf_nominal, deg=True))),
            columns=f, index=index).round(3)
        variance = pd.DataFrame(np.vstack(
            (20*np.log10(self.stft.variance()),
             np.full_like(f.values, np.NaN))),
             columns=f, index=index).round(3)

        df = pd.concat(
            (info, magnitude, phase, variance),
            keys=['info', 'magnitude_db', 'phase_deg', 'variance_db'],
            names=['type'], axis=1)

        return df

    def test(self, test_band_hz=TEST_BAND_HZ,
             max_amplitude_percent=MAX_AMPLITUDE_PERCENT,
             max_phase_degrees=MAX_PHASE_DEGREES):
        '''
        Check whether measured transfer function deviation is within
        specification, with respect to nominal.

        Parameters
        ----------
        test_band_hz : 2 element list-like of float, optional
            minimum and maxumum frequency of interest (default [0.02, 16]).
        max_amplitude_percent : float, optional
            maximum percentage deviation of amplitude (default 5).
        max_phase_degrees : float, optional
            maximum deviation of phase in degrees (default 5).

        Returns
        -------
        result : list-like of bool
            test result per channel of calibration_file
        '''
        tf_estimate = self.stft.tf_estimate()
        tf_nominal = signal.freqresp(self.lti['system'],
                                     2*np.pi*self.stft.f)[1]
        tf_deviation = tf_estimate/tf_nominal

        self.info['spec_min_freq_hz'] = test_band_hz[0]
        self.info['spec_max_freq_hz'] = test_band_hz[1]
        self.info['spec_max_amp_pct'] = max_amplitude_percent
        self.info['spec_max_phase_deg'] = max_phase_degrees
        self.info['in_spec'] = ~np.any(
            (np.abs(100*(np.abs(tf_deviation) - 1)) > max_amplitude_percent) &
            (np.abs(np.angle(tf_deviation, deg=True)) > max_phase_degrees) &
            (test_band_hz[0] <= self.stft.f) &
            (self.stft.f <= test_band_hz[1]), axis=1)

        get_logger(__name__).info('Result: ' + ', '.join(
            ['%s: %s' for channel, result in zip(
                [trace.id[-1] for trace in self.stream],
                ['pass' if in_spec else 'fail'
                 for in_spec in self.info['in_spec']])]))

    def write_calibrate_result(self):
        '''
        Write IMS2.0 CALIBRATE_RESULT message to file with data in FAP2 format.
        '''
        keep = ((self.stft.f >= self.info['spec_min_freq_hz']) &
                (self.stft.f <= self.info['spec_max_freq_hz']))
        frequencies = self.stft.f[keep]
        if len(frequencies) >= MAX_FAP_LEN:
            get_logger(__name__).warning(
                '%d frequencies is more than %d supported by FAP2 format' %
                (len(frequencies), MAX_FAP_LEN))
        tf_estimate = (
            self.stft.tf_estimate()[:, keep] /
            signal.freqresp(self.lti['cal'], 2*np.pi*frequencies)[1])

        output_txt = 'calibrate_result_{station}_{start}.txt'.format(
            station=self.stream[0].stats.station,
            start=self.info['start'].strftime('%Y%m%d.%H%M'))
        get_logger(__name__).info(output_txt)

        with open(output_txt, 'w') as file:
            file.write(IMS_HEADER.substitute(
                msg_id=CAL_RESULT_MSG_ID.format(
                    year=self.info['start'].year,
                    station=self.stream[0].stats.station),
                ref_id='XXXXXXXXXXXX',
                time_stamp=UTCDateTime().strftime(IMS_DATETIME_FMT)))

            for trace, amplitudes, phases, in_spec in zip(
                    self.stream, np.abs(tf_estimate),
                    np.angle(tf_estimate, deg=True),
                    self.info['in_spec']):
                file.write(RESPONSE_HEADER.substitute(
                    station=trace.stats.station,
                    channel=trace.stats.channel,
                    in_spec='YES' if in_spec else 'NO'))
                file.write(FAP_HEADER.format(
                    stage=1,
                    units=self._sensor_stage().output_units,
                    decimation='',
                    group_correction=0,
                    count=len(frequencies),
                    description=('Input units: ' +
                                 self._sensor_stage().input_units.lower())))
                for frequency, amplitude, phase in zip(
                        frequencies, amplitudes, phases):
                    file.write(FAP_DATA.format(
                        frequency=frequency,
                        amplitude=amplitude,
                        phase=phase))

            file.write(IMS_FOOTER)

    def simulate_response(self, trim=True):
        '''
        Simulate nominal response of sensor to calibration signal
        '''
        nominal_paz = {
            'zeros': self.lti['system'].zeros,
            'poles': self.lti['system'].poles,
            'sensitivity': abs(self.lti['system'].freqresp(w=2*np.pi)[1])}
        nominal_paz['gain'] = (self.lti['system'].gain /
                               nominal_paz['sensitivity'])

        signal = simulate_seismometer(
            self.input_voltage(trim=False), self.sampling_rate(),
            paz_simulate=nominal_paz, simulate_sensitivity=True)

        if trim:
            num_start = int(self.pre_seconds()*self.sampling_rate())
            num_end = int(self.post_seconds()*self.sampling_rate())
            signal = signal[num_start:-num_end]

        return signal

    def estimate_errors(self, variance_threshhold=0.01):
        '''
        Least-squares estimation of gain and timing errors.

        The optimal estimate of the error delta_gain is simply the average
        over frequency weighted by the estimated variance at each frequency.
        Alternately we have B=gain [Mx1] and desire to find x=gain_error
        [scalar] which is optimal over all frequencies so we set A=1 [Mx1]
        and use readily available tools to minimize the Euclidean 2-norm
        ||B - A*x||^2, a standard problem of linear algebra.

        The random error in the estimated gain is
        (Bendat & Piersol, 1993, eq. 11.55, p. 307):
            gain_variance = (1/gamma^2 - 1)/(2*n)
        Where gamma is the coherence and n is the number of statistically
        independent windows. The optimal weight is the inverse square root of
        the variance.

        A time delay between input and output of delta_t [s] results in an
        apparent phase shift, delta_theta [radians] which varies linearly
        with angular frequency w [radians/s]:
            delta_theta = delta_t * w
        We have m measurements of the phase shift B=delta_theta [Mx1 radians]
        at each frequency A=w [Mx1 radians/s], and we want to find
        x=delta_t [scalar s] which minimizes the Euclidean 2-norm
        ||B - A*x||^2, a standard problem of linear algebra.

        When the gain variance is sufficiently small, the random error in the
        expected phase, measured in radians, is actually the same as for the
        gain_variance (Bendat & Piersol, 1993, eq. 11.58, p. 309)
            phase_variance = (1/gamma^2 - 1)/(2*n) [radians]
        This is the optimal weight for a least-squares solution.

        Although this procedure could be modified to compute estimates of
        errors on a per-channel basis, it instead treats the errors as common
        to all channels.

        Parameters
        ----------
        confidence: float, optional
            confidence interval for estimates of error in results

        Returns
        -------
        gain_error, gain_error_std, time_error, time_error_std: float
            least-squares estimate of the overall errors, and estimates of the
            error in those estimates with the specified confidenc
        summary: string
            summary of fitting results
        '''
        logger = get_logger(__name__)
        f = self.stft.f
        tf_estimate = self.stft.tf_estimate()
        with np.errstate(divide='ignore', invalid='ignore'):
            tf_estimate /= signal.freqresp(self.lti['system'], 2*np.pi*f)[1]

        variance = self.stft.variance()
        if variance_threshhold is not None:
            keep = (variance < variance_threshhold).any(axis=0)
        else:
            keep = np.full_like(variance, True)
        keep[0] = False
        tf_estimate = tf_estimate[:, keep]
        variance = variance[:, keep]
        f = f[keep]
        logger.info(
            'Discarding %d points with variance > %g while fitting'
            % (keep.sum(), variance_threshhold))

        if len(f) < 1:
            logger.error('No data to fit.')
            return None, None, None, None, ''

        magnitude = np.abs(tf_estimate)
        phase = unwrap_mid(np.angle(tf_estimate), f, axis=1)

        f = np.reshape(np.tile(f, magnitude.shape[0]), (-1, 1))
        magnitude = np.reshape(magnitude, (-1, 1))
        phase = np.reshape(phase, (-1, 1))
        variance = np.reshape(variance, (-1, 1))
        weights = np.sqrt(1/variance)

        self.fit.gain = sm.WLS(magnitude, np.ones_like(f), weights).fit()
        self.fit.timing = sm.WLS(phase, 2*np.pi*f, weights).fit()

    def _save_image(self, fig=None, option_list=None):
        '''
        Save a figure with an automatically descriptive file name.
        '''
        if fig is None:
            fig = plt.gcf()
        if isinstance(option_list, str):
            option_list = option_list.split(',')

        # pylint:disable=protected-access
        caller_name = sys._getframe(1).f_code.co_name
        plot_type = caller_name.replace('plot_', '')

        file_parts = [plot_type]

        if option_list is not None:
            file_parts += [option for option in option_list if option]

        if self.stream is not None:
            start_string = short_utc(np.max([trace.stats.starttime
                                             for trace in self.stream]))
            start_string = (start_string.replace(' ', '.').replace(':', '')
                            .replace('-', ''))

            common_name = factor_names(self.stream)[0]

            file_parts += [common_name, start_string]

        THIS_FILE_NAME = '_'.join(file_parts) + '.png'

        logger = get_logger(__name__)
        logger.info(THIS_FILE_NAME)
        plt.savefig(THIS_FILE_NAME, dpi=300, bbox_inches='tight')

    def plot_check(self, where='start', window_seconds=5):
        '''
        Spot check critical times in the calibration
        '''
        assert where in ['start', 'end', 'on', 'off']

        if where == 'on':
            target_time = self.stream[0].stats.starttime + window_seconds/2
        elif where == 'off':
            target_time = self.stream[0].stats.endtime - window_seconds/2
        else:
            target_time = self.info[where]

        stream = self.stream.copy().trim(target_time - window_seconds/2,
                                         target_time + window_seconds/2)
        fig = stream.plot(handle=True, equal_scale=False)
        if where in ['start', 'end']:
            for ax in fig.axes:
                ax.axvline(target_time.datetime, linestyle='--', linewidth=0.5,
                           label=where)
        fig.axes[-1].legend(loc='lower left')

        if self.savefig:
            self._save_image(option_list=where)

    # pylint: disable=arguments-differ
    def plot_stream(self, which='output', trim=True):
        '''
        Quick plot of calibration input
        '''
        self.get_stream(which=which, trim=trim).plot(handle=True,
                                                     method='full')

        if self.savefig:
            self._save_image(option_list=[which, 'trimmed'*trim])

    def plot_response(self, model='system', f_limits=None):
        '''
        Plot nominal transfer function between specified frequency limits.

        Arguments
        ---------
        model: str, optional
            'sensor' for the sensor itself,
            'cal' for calibration input or
            'system' for the combination of the two
        f_limits: tuple, optional
            minimum and maximum frequencies for plotting, defaults to range
            set by :func:`self.compute`
        save: bool, optional
            whether to save plot as PNG
        '''
        logger = get_logger(__name__)
        if model not in self.lti.keys():
            logger.warning(
                "'%s' not among supported models: %s."
                % (model, ', '.join("'%s'" % key for key in self.lti)))
            return

        if f_limits is None:
            f_limits = [(self.stft.f)[1], (self.stft.f)[-1]]
        f_plot = logspace(f_limits[0], f_limits[1])
        tf_model = signal.freqresp(self.lti[model], 2*np.pi*f_plot)[1]

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width, width))
        axes[0].semilogx(f_plot, 20*np.log10(np.abs(tf_model)), label=model)
        axes[1].semilogx(f_plot, np.angle(tf_model, deg=True), label=model)
        axes[0].axvline(self.sampling_rate()/2, linestyle='--', color='0.5')
        axes[1].axvline(self.sampling_rate()/2, linestyle='--', color='0.5')
        if np.any(np.abs(axes[1].get_ylim()) > 180):
            axes[1].set_ylim([-180, 180])
            axes[1].set_yticks(np.arange(-180, 180 + 1, 45.))

        if model == 'cal':
            axes[0].set_ylabel('Gain [dB wrt %s/(%s)]' % (
                self._sensor_stage().input_units.lower(),
                self._sensor_stage().output_units))
        elif model == 'sensor':
            axes[0].set_ylabel('Gain [dB wrt %s/(%s)]' % (
                self._sensor_stage().output_units,
                self._sensor_stage().input_units.lower()))
        else:
            axes[0].set_ylabel('Gain [dB]')
        axes[0].legend(loc='best')
        axes[1].set_ylabel('Phase [°]')
        axes[1].set_xlabel('Frequency [Hz]')
        subplots_squeeze(fig, hspace=0)

        if self.savefig:
            self._save_image(fig, model)

    def plot_simulated(self, trim=True):
        '''
        Plot simulated calibration response in time domain.
        '''
        simulated = self.simulate_response(trim=trim)

        fig, ax = plt.subplots()
        if self.stream is not None:
            labels = factor_names(self.stream)[1]
            for output, label in zip(self.output_voltage(), labels):
                ax.plot(np.arange(len(output))/self.sampling_rate(), output,
                        label=label)
        ax.plot(np.arange(len(simulated))/self.sampling_rate(), simulated,
                label='simulated')
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper left')

        if self.savefig:
            self._save_image(fig)

    def plot_signal_to_noise(self):
        '''
        Plot estimated signal-to-noise ratio.
        '''
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]
        coherence_squared = self.stft.coherence_squared()

        fig, ax = plt.subplots()
        for snr, label in zip(10*np.log10(1/(1 - coherence_squared)), labels):
            ax.semilogx(self.stft.f, snr, label=label)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('SNR [dB]')
        ax.legend(loc='upper left')

        if self.savefig:
            self._save_image(fig)

    def plot_spectrogram(self):
        '''
        Plot input and output spectrograms, referred to sensor input [V].
        '''
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]

        # convert to volts and make stackable
        f = self.stft.f
        t = self.stft.t
        tf_system = signal.freqresp(self.lti['system'], 2*np.pi*f)[1]
        tf_system = tf_system.reshape((1, -1, 1))
        p_dbs = 10*np.log10(np.concatenate(
            (self.stft.p_xx.reshape(-1, len(f), len(t)),
             self.stft.p_yy/np.abs(tf_system)**2), axis=0))
        max_db = p_dbs.max()

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(len(labels), 1, sharex=True,
                                 figsize=(width, len(labels)*width/3))
        for p_db, ax, label in zip(p_dbs, axes, labels):

            image = ax.pcolormesh(t, f, p_db, vmin=max_db - 100, vmax=max_db)
            ax.set_ylabel('Frequency [Hz]')
            ax.text(0.95, 0.9, label, transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(facecolor='white'))

        axes[-1].set_xlabel('Time [s]')
        subplots_squeeze(fig, hspace=0)

        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7,
                     label='Power Spectral Density [dB wrt $V^2/Hz$]')

        if self.savefig:
            self._save_image(fig)

    def plot_variance(self, scale='log'):
        '''
        Plot variance on log or linear scale.

        Parameters
        ----------
        scale: str, optional
            selects 'log' or 'linear' scaling for x-axis
        '''

        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]

        variances = self.stft.variance()

        fig, ax = plt.subplots()
        for snr, label in zip(variances, labels):
            ax.semilogx(self.stft.f, snr, label=label)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Relative Transfer Function Variance')
        ax.set_yscale(scale)
        ax.legend(loc='upper left')

        if self.savefig:
            self._save_image(fig)

    def plot_transfer_function(self, scale='log', model='system',
                               remove_nominal=True, treat_errors=None,
                               save=False, confidence=0.95,
                               variance_threshhold=None):
        '''
        Plot calibration transfer function on log or linear scale.

        Optional smoothing is done using
        :func:`~obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing`.

        Parameters
        ----------
        scale: str, optional
            selects 'log' or 'linear' scaling for x-axis
        '''
        logger = get_logger(__name__)
        if model not in self.lti.keys():
            logger.warning(
                "'%s' not among supported models: %s."
                % (model, ', '.join("'%s'" % key for key in self.lti)))
            return

        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        option_list = []
        labels = factor_names(self.stream)[1]
        if remove_nominal:
            labels = [
                label + ': ' + 'pass' if in_spec else 'fail'
                for label, in_spec in zip(labels, self.info['in_spec'])]
        tf_estimate = self.stft.tf_estimate()
        variance = self.stft.variance()

        f = self.stft.f

        if remove_nominal:
            tf_remove = signal.freqresp(self.lti[model], 2*np.pi*f)[1]
            with np.errstate(divide='ignore', invalid='ignore'):
                tf_estimate /= tf_remove
            option_list += ['nominal_' + model + '_removed']
        else:
            tf_remove = np.ones(f.shape)

        with np.errstate(divide='ignore', invalid='ignore'):
            tf_nominal = signal.freqresp(self.lti['system'],
                                         2*np.pi*f)[1]/tf_remove

        if treat_errors in ['estimate', 'correct']:

            if self.fit.gain is None or self.fit.timing is None:
                raise RuntimeError('Run estimate_errors() first.')

            tf_error = tf_nominal*np.exp(1j*2*np.pi*f*self.fit.timing.params)
            tf_error *= self.fit.gain.params
        if treat_errors == 'correct':
            with np.errstate(divide='ignore', invalid='ignore'):
                tf_estimate /= self.fit.gain.params
            tf_estimate /= np.exp(1j*2*np.pi*f*self.fit.timing.params)

        if variance_threshhold is not None:
            tf_estimate[variance > variance_threshhold] = np.nan
            option_list += ['variance_lt_%g' % variance_threshhold]

        tf_magnitude = 20*np.log10(np.abs(tf_estimate))
        tf_phase = np.angle(tf_estimate, deg=True)

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width, width))

        for gain, label in zip(tf_magnitude, labels):
            axes[0].plot(f, gain, label=label)
        axes[0].set_xlim((f[1], f[-1]))
        gain_nominal = 20*np.log10(np.abs(tf_nominal))
        if not np.allclose(gain_nominal, 0):
            axes[0].plot(f, gain_nominal, label='nominal')
        axes[0].set_ylim((floor(gain_nominal[1:].min()) - 1,
                          ceil(gain_nominal[1:].max()) + 1))
        if treat_errors == 'estimate':
            axes[0].plot(f, 20*np.log10(np.abs(tf_error)),
                         label='error')

        for phase, label in zip(tf_phase, labels):
            axes[1].plot(f, phase, label=label)

        axes[1].set_xlabel('Frequency [Hz]')
        phase_nominal = unwrap_mid(np.angle(tf_nominal), f)*180./np.pi

        if not np.allclose(phase_nominal, 0):
            axes[1].plot(f, phase_nominal, label='nominal')
        axes[1].set_ylim((floor(phase_nominal.min()) - 10,
                          ceil(phase_nominal.max()) + 10))
        if treat_errors == 'estimate':
            axes[1].plot(f, np.angle(tf_error, deg=True),
                         label='error')

        band_hz = (self.info['spec_min_freq_hz'],
                   self.info['spec_max_freq_hz'])
        max_mag_db = 20*np.log10((1 + self.info['spec_max_amp_pct']/100))
        max_phase_deg = self.info['spec_max_phase_deg']
        ids_start = '\n'.join([
            factor_names(self.stream)[0],
            self.info['start'].strftime('%Y-%m-%d %H:%M')])
        test_limits = '±%g%% & ±%g°\n%g—%g Hz' % (
            (self.info['spec_max_amp_pct'], self.info['spec_max_phase_deg']) +
            band_hz)
        axes[0].annotate(ids_start, (0.025, 0.025), xycoords='axes fraction',
                         ha='left', va='bottom')
        spec = (f >= band_hz[0]) & (f <= band_hz[1])
        axes[0].fill_between(f[spec],
                             gain_nominal[spec] - max_mag_db,
                             gain_nominal[spec] + max_mag_db,
                             color='0.5', alpha=0.5, label=test_limits)
        axes[1].fill_between(f[spec],
                             phase_nominal[spec] - max_phase_deg,
                             phase_nominal[spec] + max_phase_deg,
                             color='0.5', alpha=0.5, label=test_limits)

        if remove_nominal and model == 'system':
            axes[0].set_ylabel('Gain wrt nominal [dB]')
        elif remove_nominal and model == 'cal':
            axes[0].set_ylabel('Gain [dB wrt %s/(%s)]' % (
                self._sensor_stage().output_units,
                self._sensor_stage().input_units.lower()))
        else:
            axes[0].set_ylabel('Gain [dB]')
        axes[1].set_ylabel('Phase [°]')

        if treat_errors in ['estimate', 'correct']:
            axes[0].annotate(self.fit.gain_summary(), (0.025, 0.95),
                             xycoords='axes fraction', ha='left', va='top')
            axes[1].annotate(self.fit.timing_summary(), (0.025, 0.95),
                             xycoords='axes fraction', ha='left', va='top')

        axes[0].set_xscale(scale)
        if scale != 'log':
            option_list += [scale]

        axes[0].legend(loc='lower right')
        subplots_squeeze(fig, hspace=0)

        if self.savefig:
            self._save_image(fig, option_list)


def calibration_analyzer(pattern=DEFAULT_OUTPUT_PATTERN,
                         num_windows=DEFAULT_NUM_WINDOWS, len_fft=None,
                         window=DEFAULT_WINDOW,
                         response_pattern=DEFAULT_RESPONSE_PATTERN,
                         calibration_file=DEFAULT_CALIBRATION_FILE,
                         delay_start=DEFAULT_DELAY_START,
                         write_ims=False, test_band_hz=TEST_BAND_HZ,
                         test_limits=(MAX_AMPLITUDE_PERCENT,
                                      MAX_PHASE_DEGREES),
                         plot=False, diagnostic=False, dpi=DEFAULT_DPI):

    pattern_slug = ''.join(char for char in os.path.splitext(pattern)[0]
                           if char.isalnum())
    output_parts = [os.path.splitext(THIS_FILE_NAME)[0]]
    if pattern_slug:
        output_parts += pattern_slug.split('_')
    summary_csv = '_'.join(output_parts) + '.csv'

    analyzer = CalibrationAnalyzer(savefig=plot or diagnostic)
    logger = get_logger(__name__)
    if os.path.exists(summary_csv) and os.path.isfile(summary_csv) and \
            not os.access(summary_csv, os.W_OK):
        logger.error('Will not be able to write summary to %s.' % summary_csv)
        return ''

    output_files = sorted([item for item in glob(pattern)])
    if not output_files:
        analyzer.logger.error('No files match pattern "%s".' % pattern)
        return ''

    dfs = []
    for output_file in output_files:
        plt.close('all')

        analyzer.load_stream(output_file, adjust_start=0)
        analyzer.load_response(response_pattern)
        analyzer.setup_nominal_response()
        analyzer.load_calibration(calibration_file)

        if plot or diagnostic:
            analyzer.plot_check('start')

        if diagnostic:
            analyzer.plot_check('end')

        analyzer.compute(num_windows=num_windows, len_fft=len_fft or None)
        analyzer.test(test_band_hz=test_band_hz,
                      max_amplitude_percent=test_limits[0],
                      max_phase_degrees=test_limits[1])

        if diagnostic:
            analyzer.estimate_errors()

        if plot or diagnostic:
            analyzer.plot_transfer_function()
            analyzer.plot_transfer_function(model='cal')

        if diagnostic:
            analyzer.plot_transfer_function(remove_nominal=False)
            analyzer.plot_transfer_function(scale='linear',
                                            treat_errors='estimate',
                                            variance_threshhold=0.01)
            analyzer.plot_transfer_function(scale='log',
                                            treat_errors='correct')

            analyzer.plot_signal_to_noise()

            analyzer.plot_response('sensor', f_limits=[1e-3, 1e3])
            analyzer.plot_response('system')
            analyzer.plot_response('cal')

            analyzer.plot_simulated()

            analyzer.plot_spectrogram()

        if plot:
            analyzer.plot_variance()

        dfs.append(analyzer.summary())

        if write_ims:
            analyzer.write_calibrate_result()

    if not dfs:
        logger.error('No valid calibration results.')
        return ''

    df = pd.concat(dfs)

    if len(dfs) > 0:  # drop redundant nominal response rows
        df = df.loc[~df.index.duplicated(keep='last')]

    logger.info('Summary: ' + summary_csv)
    # transpose rows and columns before writing to file
    df.reset_index(col_level=1, col_fill='info').T.to_csv(summary_csv)
    return summary_csv


def main(argv=None):
    '''
    Return zero for successful termination, one otherwise.
    '''
    if argv is None:
        argv = sys.argv
    parser = _argparser()
    args = parser.parse_args(argv[1:])

    config = vars(args).copy()

    result = calibration_analyzer(**config)

    return len(result) == 0


if __name__ == '__main__':
    sys.exit(main())
