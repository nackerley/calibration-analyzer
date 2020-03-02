# -*- coding: utf-8 -*-
"""
A collection of functions useful for seismograph calibration.
"""
# pylint: disable=logging-not-lazy
from __future__ import absolute_import, division, print_function

import os
import lzma
import logging
from warnings import warn
from operator import mul
from functools import reduce
from struct import calcsize
from math import log10, floor, ceil
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.special import erfinv
import statsmodels.api as sm

from obspy import read, read_inventory, Trace, Stream
from obspy.signal.invsim import simulate_seismometer
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

from catalogue_tools.utilities import (
    pretty_duration, pretty_voltage, pretty_bytes,
    parse_duration, parse_voltage, round_sig,
    logspace, get_logger)

from calan.core import (
    Stft, factor_names, subplots_squeeze, inventory_items,
    lti_from_zpsf, minreal, unwrap_mid, truncnorm_shape,
    len_fft_welch, num_windows_welch, fft_frequencies,
    extract_decimation_coefficients, compute_decim_delay, multi_decim)
from calan.stream_analyzer import StreamAnalyzer

# %% constants
FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(FILE_NAME)[0] + '.log'

# defaults
DEFAULT_LEAD_IN = 600
DEFAULT_LEAD_OUT = 600
DEFAULT_PRE_TIME = 10
DEFAULT_POST_TIME = 10

# Nanometrics Centaur User Guide 17935R5, 2016-11-02
CALIBRATION_SAMPLE_RATE = 30e3
FULL_SCALE_VOLTAGE = 5
DAC_BITS = 16
DAC_GAIN = FULL_SCALE_VOLTAGE/(2**(DAC_BITS - 1) - 1)
SAMPLE_FORMAT = '<h'  # little-endian short (16-bit) integer
CALIBRATION_DTYPE = 'int16'

# Nanometrics Trillium 120Q/QA User Guide 17751R2, 2015-10-05
CAL_ZEROS = ()  # rad/s
CAL_POLES = (-160, -3177)  # rad/s
CAL_SENS = -0.009715  # m/s^2/V
CAL_FREQ = 0.001  # Hz

# other handy constants
NP_DTYPE = np.dtype('int%d' % DAC_BITS)


# %% definitions

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


def expected_wav_size(duration):
    '''
    Compute expected size before compression.
    '''
    return pretty_bytes(
        calcsize(SAMPLE_FORMAT)*duration*CALIBRATION_SAMPLE_RATE)


def make_random_signal_file_name(
        style, duration_seconds, pp_voltage=0, rms_voltage=0, mean_voltage=0,
        lead_in=0, lead_out=0, sample_rate=100):
    '''
    Generate a gaussian white noise calibration signal file name.
    '''

    file_name = ''
    if style != '':
        file_name += style + '_'
    if rms_voltage != 0:
        file_name += '%srms' % pretty_voltage(rms_voltage)
    if pp_voltage != 0:
        file_name += '%spp' % pretty_voltage(pp_voltage)
    file_name += '_%s' % pretty_duration(duration_seconds)
    if sample_rate != CALIBRATION_SAMPLE_RATE:
        file_name += '_%gsps' % sample_rate
    if lead_in != 0:
        file_name += '_on%s' % pretty_duration(lead_in)
    if lead_out != 0:
        file_name += '_off%s' % pretty_duration(lead_out)
    if mean_voltage != 0:
        file_name += '_mean%s' % pretty_voltage(lead_in)

    return file_name


def parse_random_signal_file_name(file_name):
    '''
    Parse a gaussian white noise calibration signal file name.

    :rtype: tuple
    :returns: style, duration_seconds, pp_voltage, rms_voltage, mean_voltage,
            lead_in, lead_out, sample_rate
    '''

    file_name = file_name.replace('.wav', '').replace('.gz', '')
    parts = os.path.split(file_name)[1].split('_')

    if not any(char.isdigit() for char in parts[0]):
        style = parts.pop(0)
    else:  # for backward compatibility
        style = ''
    duration_seconds = float(parse_duration(parts.pop(1)))

    extras = []
    for token, parser, default in zip(
            ['pp', 'rms', 'mean', 'on', 'off', 'sps'],
            [parse_voltage, parse_voltage, parse_voltage,
             parse_duration, parse_duration, float],
            [0, 0, 0, 0, 0, CALIBRATION_SAMPLE_RATE]):
        index = next((i for i, part in enumerate(parts)
                      if token in part), None)
        if index is not None:
            extras.append(parser(parts.pop(index).replace(token, '')))
        else:
            extras.append(default)

    if parts:
        warn('Not parsed: %s' % '_'.join(parts))

    # pylint:disable=unbalanced-tuple-unpacking
    (pp_voltage, rms_voltage, mean_voltage, lead_in, lead_out,
     sample_rate) = extras

    return (style, duration_seconds, pp_voltage, rms_voltage, mean_voltage,
            lead_in, lead_out, sample_rate)


def _add_turn_on_off_times(signal, lead_in, lead_out):
    len_on = int(round(lead_in*CALIBRATION_SAMPLE_RATE))
    len_off = int(round(lead_out*CALIBRATION_SAMPLE_RATE))
    return np.concatenate(
        (np.zeros((len_on, )), signal, np.zeros((len_off, ))))


def generate_gaussian(duration_seconds, rms_voltage, mean_voltage=0,
                      lead_in=300, lead_out=300,
                      sample_rate=CALIBRATION_SAMPLE_RATE):
    '''
    Generate a gaussian white noise calibration signal.

    A random seed is hard-coded so that for a given set of inputs the same
    signal will always be generated.

    The output sample_rate effectively acts as a low-pass filter and gives the
    resulting signal good compressibility, since most of the first-differences
    will be zero. Note that duration_seconds is forced to a multiple of
    1/sample_rate and does not include lead_in and lead_out (also in seconds).
    '''
    logger = logging.getLogger(__name__)
    file_name = make_random_signal_file_name(
        'gaussian', duration_seconds, rms_voltage=rms_voltage,
        sample_rate=sample_rate, mean_voltage=mean_voltage,
        lead_in=lead_in, lead_out=lead_out)
    file_size = expected_wav_size(duration_seconds + lead_in + lead_out)
    logger.debug('Uncompressed output "%s.wav" will be %s.'
                 % (file_name, file_size))

    # generate random signal
    shape_a, shape_b = truncnorm_shape(mean_voltage, rms_voltage,
                                       FULL_SCALE_VOLTAGE)
    length = int(round(duration_seconds*sample_rate))
    signal_volts = stats.truncnorm.rvs(shape_a, shape_b,
                                       size=length, random_state=42)

    # stretch signal to calibration circuit output sample rate
    signal_volts = np.repeat(signal_volts,
                             int(CALIBRATION_SAMPLE_RATE/sample_rate))

    signal_volts = _add_turn_on_off_times(signal_volts, lead_in, lead_out)

    return signal_volts, file_name


def generate_random_binary(duration_seconds, pp_voltage, sample_rate,
                           lead_in=300, lead_out=300):
    '''
    Generate a random binary calibration signal.

    A random seed is hard-coded so that for a given set of inputs the same
    signal will always be generated.

    Note that duration_seconds is forced to a multiple of
    1/sample_rate and does not include lead_in and lead_out (also in seconds).
    '''
    logger = logging.getLogger(__name__)
    file_name = make_random_signal_file_name(
        'binary', duration_seconds, pp_voltage=pp_voltage,
        sample_rate=sample_rate, lead_in=lead_in, lead_out=lead_out)
    file_size = expected_wav_size(duration_seconds + lead_in + lead_out)
    logger.debug('Uncompressed output "%s.wav" will be %s.'
                 % (file_name, file_size))

    # generate random binary signal and scale it appropriately
    length = int(round(duration_seconds*sample_rate))
    signal_volts = stats.randint.rvs(0, 2, size=(length, ), random_state=42)
    signal_volts = pp_voltage*(signal_volts.astype(float) - 0.5)

    # stretch signal to calibration circuit output sample rate
    signal_volts = np.repeat(signal_volts,
                             int(CALIBRATION_SAMPLE_RATE/sample_rate))

    signal_volts = _add_turn_on_off_times(signal_volts, lead_in, lead_out)

    return signal_volts, file_name


def generate_piecewise_constant(durations, voltages):
    '''
    Generate a piecewise constant calibration signal in volts.
    '''
    logger = logging.getLogger(__name__)
    durations = np.asarray(durations)
    voltages = np.asarray(voltages)

    file_name = 'step_%s' % '_'.join(
        ['%gV_%ss' % (voltage, duration)
         for voltage, duration in zip(voltages, durations)])
    file_size = expected_wav_size(durations.sum())
    logger.debug('Uncompressed output "%s.wav" will be %s.'
                 % (file_name, file_size))

    times = durations.cumsum()
    t = np.arange(0, durations.sum(), 1/CALIBRATION_SAMPLE_RATE).reshape(-1, 1)
    indices = np.argmax(t < times, axis=1)

    return voltages[indices], file_name


def plot_calibration(signal,
                     units='counts', file_name=None, sample_rate=100):
    '''
    Plot a calibration signal, decimated to given sample rate.

    CAUTION: decimation is performed without prior filtering, so resulting
    signal may be strongly aliased.
    '''
    t = np.arange(len(signal))/CALIBRATION_SAMPLE_RATE

    step = int(CALIBRATION_SAMPLE_RATE/sample_rate)
    _, ax = plt.subplots()
    ax.plot(t[::step], signal[::step], marker='x')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal [%s]' % units)

    if file_name is not None:
        file_name = os.path.splitext(file_name)[0] + '.png'
        plt.gcf().savefig(file_name, dpi=300, bbox_inches='tight')


def _chunk_fmt(len_chunk):
    return '%s%d%s' % (SAMPLE_FORMAT[0], len_chunk, SAMPLE_FORMAT[1])


def get_times(signal, b_stages=None, factors=None, discard_initial=True):
    '''
    Generate an array of times corresponding to a calibration signal.
    '''
    i = np.arange(len(signal))
    if factors:
        i = i*np.prod(factors)
    if factors and b_stages:
        if discard_initial:
            i = i + compute_decim_delay(b_stages, factors)/2
        else:
            i = i - compute_decim_delay(b_stages, factors)/2
    return i/CALIBRATION_SAMPLE_RATE


def pad_for_decimation(signal, b_stages, factors):
    '''
    Pad a signal with zeros so that the first sample after decimation will be
    at the same time as the first sample before decimation.
    '''
    n_pad_upsample = compute_decim_delay(b_stages, factors)
    signal_padded = np.hstack((np.zeros((int(n_pad_upsample/2), )), signal,
                               np.zeros((int(n_pad_upsample/2), ))))
    t_start = -n_pad_upsample/2/CALIBRATION_SAMPLE_RATE
    return signal_padded, t_start


def sample_hold_digitize(signal):
    '''
    Process signal as if sampled and held by a DAC, then digitized by an ADC.
    '''
    signal = np.concatenate((np.zeros((1, )), signal, np.zeros((1, ))))
    signal = signal[:-1]/2 + signal[1:]/2
    return signal


def plot_calibration_decimated(sig_in, sig_out, b_stages, factors,
                               discard_initial):
    '''
    Utility for comparing timing of signals before/after decimation.
    '''

    t_in = get_times(sig_in)
    t_out = get_times(sig_out, b_stages=b_stages, factors=factors,
                      discard_initial=discard_initial)

    _, ax = plt.subplots()
    ax.plot(t_in, sig_in, label='input')
    ax.plot(t_out, sig_out, marker='x', label='output')
    ax.set_xlabel('Time [s]')
    ax.grid()
    ax.legend()


class CalibrationAnalyzer(StreamAnalyzer):
    '''
    A calibration signal analyzer based on :class:`obspy.Stream`.

    TODO:
     * Remove ability to request signals from FDSNWS; must provide files
     * Introduce default lead-in/lead-out/turn-on/turn-off times and use them
       to infer start_time from data
     * add CLI
     * Reuse decimated version of calibration signal for multiple calibrations
     * Make HHC channel only when it doesn't exist & cache as 4-channel stream
     * Make savefig a class attribute
     * require calibration_file and gain only upon load_calibration()
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
        get_logger(self.__class__.__name__, LOG_FILE_NAME, log_level)

        self.stream = Stream()
        self.info = {
            'waveform_file': '',
            'inventory_file': '',
            'calibration_file': '',
            'start': None,
            'end': None}

        self.lti = {'cal': None,
                    'sensor': None,
                    'system': None}
        self.stft = Stft()
        self.savefig = savefig

    def load_stream(self, waveform_file=None, lead_in=DEFAULT_LEAD_IN,
                    lead_out=DEFAULT_LEAD_OUT, pre_time=DEFAULT_PRE_TIME,
                    post_time=DEFAULT_POST_TIME):
        '''
        Load calibration stream.

        Two cases are supported:
            1.  The stream does not include a calibration signal:
                The calibration start and end are inferred from lead
                in/out and event pre/post times.
            2.  The stream includes a calibration signal:
                The calibration start and end are that of the stream.
        '''
        logger = get_logger(__name__)
        logger.info(waveform_file)
        self.stream = read(waveform_file)
        self.info['waveform_file'] = waveform_file
        self.info['start'] = self.stream[0].stats.starttime
        self.info['end'] = self.stream[0].stats.endtime

        if not any(trace.id[-1] == 'C' for trace in self.stream):
            self.info['start'] += lead_in + pre_time
            self.info['end'] -= lead_out + post_time

    def sampling_rate(self):
        '''Return stream sampling rate'''

        return self.stream[0].stats.sampling_rate

    def pre_seconds(self):
        '''Return sum of lead-in and event pre-time.'''

        return self.info['start'] - self.stream[0].stats.starttime

    def post_seconds(self):
        '''Return sum of lead-out and event post-time'''

        return self.stream[0].stats.endtime - self.info['end']

    def load_response(self, inventory_file, ignore_open_closed=True,
                      force_first=False):
        '''
        Load response file describing the system being calibrated.
        '''
        logger = get_logger(__name__)
        logger.info(inventory_file)
        inventory = read_inventory(inventory_file)
        self.info['inventory_file'] = inventory_file

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

        TODO: Implement caching.
        '''
        logger = get_logger(__name__)
        if any(trace.id[-1] == 'C' for trace in self.stream):
            logger.error('Waveform data already includes calibration channel')
            return

        logger.info(calibration_file)

        # cache_file = ('%s_%gsps.%s' %
        #              (calibration_file.replace('.lzma', ''),
        #               self.stream[0].stats.sampling_rate,
        #               self.CACHE_FORMAT.lower()))
        # if os.path.exists(cache_file):
        #    logger.info(
        #        'Found cache: %s' % os.path.basename(cache_file))
        #    calibration_stream = read(cache_file)
        #    self.calibration_file
        #    return

        b_stages, factors = extract_decimation_coefficients(
            self.stream[0].stats.response.response_stages)
        factor = reduce(mul, factors)

        self.info['calibration_file'] = calibration_file
        signal = np.frombuffer(lzma.open(calibration_file).read(),
                               dtype=CALIBRATION_DTYPE)
        signal = sample_hold_digitize(signal)
        signal = pad_for_decimation(signal, b_stages, factors)[0]
        signal = self._pad_lead_in_out(signal, factor)

        logger.info('Decimating by %d ...' % factor)
        signal = multi_decim(signal, b_stages, factors)[0]

        if len(signal) != self.stream[0].stats.npts:
            logger.info(
                'Expected %d, obtained %d points for calibration signal' %
                (self.stream[0].stats.npts, len(signal)))
            logger.error('Check lead-in and lead-out times.')
            import pdb; pdb.set_trace()
            return

        stats = {key: self.stream[0].stats[key] for key in
                 ['network', 'station', 'location', 'sampling_rate']}
        stats['channel'] = self.stream[0].stats['channel'][:2] + 'C'
        stats['starttime'] = self.stream[0].stats.starttime
        trace = Trace(data=np.ascontiguousarray(signal), header=stats)
        self.stream.traces.append(trace)

        # logger.info(
        #     'Caching: %s' % os.path.basename(cache_file))
        # self.input.write(calibration_stream, format=self.CACHE_FORMAT)

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

        stage = self.stream[0].stats.response.response_stages[0]
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
            self.stream[0].stats.response.response_stages[0].stage_gain)

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
        signal = np.row_stack([trace.data[:-1]
                               for trace in stream]).astype(float)
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
        logger = get_logger(__name__)
        x = self.input_voltage()
        y = self.output_voltage()
        f_sample = self.sampling_rate()

        if len_fft is None:
            len_fft = len_fft_welch(len(x), num_windows, fraction_overlap)
        len_overlap = int(fraction_overlap*len_fft)

        num_windows = num_windows_welch(len(x), len_fft, len_overlap)
        f_expected = fft_frequencies(len_fft, f_sample)
        logger.info(
            '%d segments ' % num_windows +
            'from %g to %g Hz' % (f_expected[1], f_expected[-1]))

        self.stft.compute(x, y, f_sample, len_fft, len_overlap)
        self.stft.trim()

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

    def estimate_errors(self, confidence=0.95, variance_threshhold=0.01):
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
        tf_estimate = self._mean(self.stft.p_xy)/self._mean(self.stft.p_xx)
        with np.errstate(divide='ignore', invalid='ignore'):
            tf_estimate /= signal.freqresp(self.lti['system'], 2*np.pi*f)[1]
        coherence_squared = np.abs(self._mean(self.stft.p_xy))**2/(
            self._mean(self.stft.p_xx)*self._mean(self.stft.p_yy))
        num_sigma = np.sqrt(2)*erfinv(confidence)

        variance = (1/coherence_squared - 1)/(2*len(self.stft.t))
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

        gain = sm.WLS(magnitude, np.ones_like(f), weights).fit()
        timing = sm.WLS(phase, 2*np.pi*f, weights).fit()

        gain_digits = round(log10(
            abs(gain.params - 1)/(num_sigma*gain.bse))) + 1
        timing_digits = round(log10(
            abs(timing.params)/(num_sigma*timing.bse))) + 1
        summary = (
            'Gain error %g%% ±%g%%\n' %
            (round_sig(100*(gain.params - 1), gain_digits),
             round_sig(100*num_sigma*gain.bse, 1)) +
            'Timing error %s ±%s\n' %
            (pretty_duration(timing.params, fmt=timing_digits, thresh=0.05),
             pretty_duration(num_sigma*timing.bse, fmt=1, thresh=0.05)) +
            '(%.0f%% confidence)' % (100*confidence))

        return (gain.params, num_sigma*gain.bse,
                timing.params, num_sigma*timing.bse, summary)

    def plot_check(self, where='start', window_seconds=20):
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
            self.save_image(option_list=where)

    # pylint: disable=arguments-differ
    def plot_stream(self, which='output', trim=True):
        '''
        Quick plot of calibration input
        '''
        self.get_stream(which=which, trim=trim).plot(handle=True,
                                                     method='full')

        if self.savefig:
            self.save_image(option_list=[which, 'trimmed'*trim])

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

        axes[0].set_ylabel('Sensitivity [dB wrt ]')
        axes[0].legend(loc='best')
        axes[1].set_ylabel('Phase [°]')
        axes[1].set_xlabel('Frequency [Hz]')
        subplots_squeeze(fig, hspace=0)

        if self.savefig:
            self.save_image(fig, model)

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
            self.save_image(fig)

    def plot_signal_to_noise(self):
        '''
        Plot estimated signal-to-noise ratio.
        '''
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]
        coherence_squared = np.abs(self._mean(self.stft.p_xy))**2/(
            self._mean(self.stft.p_xx)*self._mean(self.stft.p_yy))

        fig, ax = plt.subplots()
        for snr, label in zip(10*np.log10(1/(1 - coherence_squared)), labels):
            ax.semilogx(self.stft.f, snr, label=label)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('SNR [dB]')
        ax.legend(loc='upper left')

        if self.savefig:
            self.save_image(fig)

    def plot_spectrogram(self):
        '''
        Plot input and output spectrograms, referred to sensor input [V].
        '''
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = ['input'] + factor_names(self.stream)[1]

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
            self.save_image(fig)

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

        coherence_squared = np.abs(self._mean(self.stft.p_xy))**2/(
            self._mean(self.stft.p_xx)*self._mean(self.stft.p_yy))
        variances = (1/coherence_squared - 1)/(2*len(self.stft.t))

        fig, ax = plt.subplots()
        for snr, label in zip(variances, labels):
            ax.semilogx(self.stft.f, snr, label=label)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Relative Transfer Function Variance')
        ax.set_ylim((0, 0.01))
        ax.set_yscale(scale)
        ax.legend(loc='upper left')

        if self.savefig:
            self.save_image(fig)

    def plot_transfer_function(self, scale='log', model='system',
                               remove_nominal=True, treat_errors='correct',
                               save=False, confidence=0.95,
                               variance_threshhold=None, smooth=False):
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
        tf_estimate = self._mean(self.stft.p_xy)/self._mean(self.stft.p_xx)

        coherence_squared = np.abs(self._mean(self.stft.p_xy))**2/(
            self._mean(self.stft.p_xx)*self._mean(self.stft.p_yy))
        variance = (1/coherence_squared - 1)/(2*len(self.stft.t))

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
            gain_error, _, time_error, _, message = \
                self.estimate_errors(confidence=confidence)
            if gain_error is None:
                return
            tf_error = tf_nominal*np.exp(1j*2*np.pi*f*time_error)
            tf_error *= gain_error
        if treat_errors == 'correct':
            with np.errstate(divide='ignore', invalid='ignore'):
                tf_estimate /= gain_error
            tf_estimate /= np.exp(1j*2*np.pi*f*time_error)

        if variance_threshhold is not None:
            tf_estimate[variance > variance_threshhold] = np.nan
            option_list += ['variance_lt_%g' % variance_threshhold]

        tf_magnitude = 20*np.log10(np.abs(tf_estimate))
        tf_phase = np.angle(tf_estimate, deg=True)

        if smooth:
            # can't smooth over nans so discard them
            keep = ~np.isnan(tf_estimate).any(axis=0)
            tf_magnitude = tf_magnitude[:, keep]
            tf_phase = tf_phase[:, keep]
            f_keep = f[keep]

            tf_magnitude = konno_ohmachi_smoothing(tf_magnitude, f_keep,
                                                   normalize=True)
            tf_phase = konno_ohmachi_smoothing(tf_phase, f_keep)
            option_list += ['smoothed']
        else:
            f_keep = f

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width, width))

        for gain, label in zip(tf_magnitude, labels):
            axes[0].plot(f_keep, gain, label=label)
        axes[0].set_ylabel('Gain [dB]')
        axes[0].set_xlim((f[1], f[-1]))
        gain_nominal = 20*np.log10(np.abs(tf_nominal[1:]))
        if not np.allclose(gain_nominal, 0):
            axes[0].plot(f[1:], gain_nominal, label='nominal')
        axes[0].set_ylim((floor(gain_nominal.min()) - 1,
                          ceil(gain_nominal.max()) + 1))
        if treat_errors == 'estimate':
            axes[0].plot(f[1:], 20*np.log10(np.abs(tf_error[1:])),
                         label='error')

        for phase, label in zip(tf_phase, labels):
            axes[1].plot(f_keep, phase, label=label)
        axes[1].set_ylabel('Phase [°]')
        axes[1].set_xlabel('Frequency [Hz]')
        phase_nominal = unwrap_mid(np.angle(tf_nominal[1:]), f[1:])*180./np.pi

        if not np.allclose(phase_nominal, 0):
            axes[1].plot(f[1:], phase_nominal, label='nominal')
        axes[1].set_ylim((floor(phase_nominal.min()) - 10,
                          ceil(phase_nominal.max()) + 10))
        if treat_errors == 'estimate':
            axes[1].plot(f[1:], np.angle(tf_error[1:], deg=True),
                         label='error')

        if treat_errors in ['estimate', 'correct']:
            axes[1].text(0.05, 0.9, message, transform=axes[1].transAxes,
                         horizontalalignment='left',
                         verticalalignment='top')

        axes[0].set_xscale(scale)
        if scale != 'log':
            option_list += [scale]

        axes[0].legend(loc='best')
        subplots_squeeze(fig, hspace=0)

        if self.savefig:
            self.save_image(fig, option_list)
