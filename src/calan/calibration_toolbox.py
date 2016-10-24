# -*- coding: utf-8 -*-
"""
A collection of functions useful for seismograph calibration.
"""

import os
import wave
import gzip
from warnings import warn
from struct import pack, unpack, calcsize
from io import BytesIO


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import lfilter, lfilter_zi

from obspy.core.inventory.response import CoefficientsTypeResponseStage

from antelope_tools.utilities import (pretty_bytes, pretty_duration,
                                      pretty_voltage,
                                      parse_duration, parse_voltage)

# Centaur calibration setup
CALIBRATION_SAMPLE_RATE = 30e3
FULL_SCALE_VOLTAGE = 5
DAC_BITS = 16
NP_DTYPE = np.dtype('int%d' % DAC_BITS)
DAC_GAIN = FULL_SCALE_VOLTAGE/(2**(DAC_BITS - 1) - 1)
SAMPLE_FORMAT = '<h'  # little-endian short (16-bit) integer
_ROOT = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


def convert_volts_to_counts(signal):
    '''
    Convert a calibration signal from volts to counts.

    See also docstring for convert_counts_to_volts.
    '''
    signal = np.array(signal, dtype='float')
    signal /= DAC_GAIN
    return signal.astype(NP_DTYPE)


def convert_counts_to_volts(signal):
    '''
    Convert a calibration signal from counts to volts.

    DAC gain is set according to statement "The maximum output signal (+5 V)
    corresponds to the maximum value (32 767)" (p. 97 of "Centaur User
    Guide 17935R5.pdf") and assumption that 0 counts gives 0 V. Thus the rest
    of table on p.97 of the user guide is in error.
    '''
    signal = np.array(signal, dtype=NP_DTYPE).astype('float')
    signal *= DAC_GAIN
    return signal


def _print_expected_file_size(file_name, duration):
    '''
    Print file name and its expected size before compression.
    '''
    file_size = pretty_bytes(
        calcsize(SAMPLE_FORMAT)*duration*CALIBRATION_SAMPLE_RATE)
    print('Uncompressed output "%s.wav" will be %s.' % (file_name, file_size))


def make_random_signal_file_name(
        style, duration_seconds, pp_voltage=0, rms_voltage=0, mean_voltage=0,
        t_on=0, t_off=0, sample_rate=100):
    '''
    Generate a gaussian white noise calibration signal file name.
    '''

    file_name = style + '_'
    if rms_voltage != 0:
        file_name += '%srms' % pretty_voltage(rms_voltage)
    if pp_voltage != 0:
        file_name += '%spp' % pretty_voltage(pp_voltage)
    file_name += '_%s' % pretty_duration(duration_seconds)
    if sample_rate != CALIBRATION_SAMPLE_RATE:
        file_name += '_%gsps' % sample_rate
    if t_on != 0:
        file_name += '_on%s' % pretty_duration(t_on)
    if t_off != 0:
        file_name += '_off%s' % pretty_duration(t_off)
    if mean_voltage != 0:
        file_name += '_mean%s' % pretty_voltage(t_on)

    return file_name


def parse_random_signal_file_name(file_name):
    '''
    Parse a gaussian white noise calibration signal file name.

    :rtype: tuple
    :returns: style, duration_seconds, pp_voltage, rms_voltage, mean_voltage,
            t_on, t_off, sample_rate
    '''

    parts = os.path.splitext(os.path.split(file_name)[1])[0].split('_')

    if not any(char.isdigit() for char in parts[0]):
        style = parts.pop(0)
        duration_seconds = parse_duration(parts.pop(2))
    else:  # for backward compatibility
        style = ''
        duration_seconds = parse_duration(parts.pop(1))

    extras = []
    for token, parser in zip(['pp', 'rms', 'mean', 'on', 'off', 'sps'],
                             [parse_voltage, parse_voltage, parse_voltage,
                              parse_duration, parse_duration,
                              lambda value: '%g' % value]):
        index = next((i for i, part in enumerate(parts)
                      if token in part), None)
        if index is not None:
            extras.append(parser(parts.pop(index).replace(token, '')))
        else:
            extras.append(0)

    if len(parts) != 0:
        warn('Portions of file name not parsed:', '_'.join(parts))

    (pp_voltage, rms_voltage, mean_voltage, t_on, t_off,
     sample_rate) = extras

    return (style, duration_seconds, pp_voltage, rms_voltage, mean_voltage,
            t_on, t_off, sample_rate)


def _add_turn_on_off_times(signal, t_on, t_off):
    len_on = int(round(t_on*CALIBRATION_SAMPLE_RATE))
    len_off = int(round(t_off*CALIBRATION_SAMPLE_RATE))
    return np.concatenate(
        (np.zeros((len_on,)), signal, np.zeros((len_off,))))


def truncnorm_shape(mean, std, clip_a, clip_b=None):
    '''
    Convert mean, standard deviation and clip levels to
    :class:`~scipy.stats.truncnorm' shape parameters.

    :returns: a, b
    '''
    if clip_b is None:
        clip_b = - clip_a
    a, b = (clip_a - mean) / std, (clip_b - mean) / std

    return a, b


def generate_gaussian(duration_seconds, rms_voltage, mean_voltage=0,
                      t_on=300, t_off=300, sample_rate=CALIBRATION_SAMPLE_RATE,
                      verbose=True):
    '''
    Generate a gaussian white noise calibration signal.

    A random seed is hard-coded so that for a given set of inputs the same
    signal will always be generated.

    The output sample_rate effectively acts as a low-pass filter and gives the
    resulting signal good compressibility, since most of the first-differences
    will be zero. Note that duration_seconds is forced to a multiple of
    1/sample_rate and does not include t_on and t_off (also in seconds).
    '''
    file_name = make_random_signal_file_name(
        'gaussian', duration_seconds, rms_voltage=rms_voltage,
        sample_rate=sample_rate, mean_voltage=mean_voltage,
        t_on=t_on, t_off=t_off)
    if verbose:
        _print_expected_file_size(file_name, duration_seconds + t_on + t_off)

    # generate random signal
    a, b = truncnorm_shape(mean_voltage, rms_voltage, FULL_SCALE_VOLTAGE)
    length = int(round(duration_seconds*sample_rate))
    signal_volts = stats.truncnorm.rvs(a, b, size=length, random_state=42)

    # stretch signal to calibration circuit output sample rate
    signal_volts = np.repeat(signal_volts,
                             int(CALIBRATION_SAMPLE_RATE/sample_rate))

    signal_volts = _add_turn_on_off_times(signal_volts, t_on, t_off)

    return signal_volts, file_name


def generate_random_binary(duration_seconds, pp_voltage, sample_rate,
                           t_on=300, t_off=300, verbose=True):
    '''
    Generate a random binary calibration signal.

    A random seed is hard-coded so that for a given set of inputs the same
    signal will always be generated.

    Note that duration_seconds is forced to a multiple of
    1/sample_rate and does not include t_on and t_off (also in seconds).
    '''
    file_name = make_random_signal_file_name(
        'binary', duration_seconds, pp_voltage=pp_voltage,
        sample_rate=sample_rate, t_on=t_on, t_off=t_off)
    if verbose:
        _print_expected_file_size(file_name, duration_seconds + t_on + t_off)

    # generate random binary signal and scale it appropriately
    length = int(round(duration_seconds*sample_rate))
    signal_volts = stats.randint.rvs(0, 2, size=(length,), random_state=42)
    signal_volts = pp_voltage*(signal_volts.astype(float) - 0.5)

    # stretch signal to calibration circuit output sample rate
    signal_volts = np.repeat(signal_volts,
                             int(CALIBRATION_SAMPLE_RATE/sample_rate))

    signal_volts = _add_turn_on_off_times(signal_volts, t_on, t_off)

    return signal_volts, file_name


def generate_piecewise_constant(durations, voltages, verbose=False):
    '''
    Generate a piecewise constant calibration signal in volts.
    '''
    durations = np.asarray(durations)
    voltages = np.asarray(voltages)

    file_name = 'step_%s' % '_'.join(
        ['%gV_%ss' % (voltage, duration)
         for voltage, duration in zip(voltages, durations)])
    if verbose:
        _print_expected_file_size(file_name, durations.sum())

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
        plt.get_figure().savefig(file_name, dpi=300, bbox_inches='tight')


def _chunk_fmt(len_chunk):
    return '%s%d%s' % (SAMPLE_FORMAT[0], len_chunk, SAMPLE_FORMAT[1])


def write_wav_gz(file_name, signal, verbose=True):
    '''
    Write a single-channel .wav.gz file with fixed encoding for Centaur
    calibration. Compression is performed in memory, not on disk. Signal must
    be in counts.
    '''
    if os.path.splitext(file_name)[1].lower() not in ['.gz', '.wav']:
        file_name += '.wav.gz'

    len_chunk = int(np.sqrt(len(signal)))
    num_chunks = int(np.ceil(float(len(signal))/len_chunk))

    with BytesIO() as stream:
        with wave.open(stream, 'w') as wave_buffer:
            wave_buffer.setnchannels(1)
            wave_buffer.setsampwidth(int(DAC_BITS/8))
            wave_buffer.setframerate(CALIBRATION_SAMPLE_RATE)

            for i_start in np.arange(num_chunks)*len_chunk:
                i_stop = i_start + len_chunk
                if i_stop > len(signal):
                    i_stop = len(signal)
                byte_array = pack(
                    _chunk_fmt(i_stop - i_start), *signal[i_start:i_stop])
                wave_buffer.writeframes(byte_array)

        stream.seek(0)
        with wave.open(stream, 'r') as wave_buffer:
            assert wave_buffer.getnframes() == len(signal)

        stream.seek(0)
        if os.path.splitext(file_name)[1].lower() == '.wav':
            with open(file_name, 'wb') as wav_file:
                wav_file.write(stream.read())
        else:
            with gzip.open(file_name, 'wb') as gz_file:
                gz_file.write(stream.read())
        wav_size = stream.tell()

    if verbose:
        out_size = os.path.getsize(file_name)
        print('Compressed size %s is %.2f%% of original size %s:\n\t%s' %
              (pretty_bytes(out_size), 100*out_size/wav_size,
               pretty_bytes(wav_size), file_name))


def read_wav_gz(file_name):
    '''
    Read a single-channel .wav or .wav.gz file with fixed encoding for Centaur
    calibration.

    FIXME: Would it be better to identify filetypes by initial bytes?
    '''
    if os.path.splitext(file_name)[1].lower() not in ['.gz', '.wav']:
        file_name += '.wav.gz'

    with BytesIO() as stream:
        if os.path.splitext(file_name)[1].lower() == '.gz':
            with gzip.open(file_name, 'rb') as gzip_file:
                stream.write(gzip_file.read())
        else:
            with open(file_name, 'rb') as wave_buffer:
                stream.write(wave_buffer.read())

        stream.seek(0)
        with wave.open(stream, 'rb') as wave_buffer:
            assert wave_buffer.getnchannels() == 1
            assert wave_buffer.getsampwidth() == int(DAC_BITS/8)
            assert wave_buffer.getframerate() == CALIBRATION_SAMPLE_RATE
            signal = np.zeros((wave_buffer.getnframes(), ))

            len_chunk = int(np.sqrt(len(signal)))
            num_chunks = int(np.ceil(float(len(signal))/len_chunk))

            for i_start in np.arange(num_chunks)*len_chunk:
                i_stop = i_start + len_chunk
                if i_stop > len(signal):
                    i_stop = len(signal)
                byte_array = wave_buffer.readframes(int(i_stop - i_start))
                signal[i_start:i_stop] = unpack(
                    _chunk_fmt(i_stop - i_start), byte_array)

    return signal


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
    signal_padded = np.hstack((np.zeros((int(n_pad_upsample/2),)), signal,
                               np.zeros((int(n_pad_upsample/2),))))
    t_start = -n_pad_upsample/2/CALIBRATION_SAMPLE_RATE
    return signal_padded, t_start


def compute_decim_delay(b_stages, factors):
    '''
    Determine total filter delay for multi-stage decimation.

    Each decimation filter must be odd-order so that the total filter delay
    is an integer number of samples. The result is the total number of extra
    samples required to load the multi decimation filters.  Actual
    filter delay, in seconds, for a symmetric filter, is:
        tDelay = n_pad_upsample/2/f_upsample
    where the initial sample rate is f_upsample in Hz.

        :param b_stages: decimation coefficients
        :type b_stages: list of :class:`~numpy.array`
        :list factors: integer decimation factors for each stage
    :returns: integer number of samples needed to load filters

    Example:
        n_pad_upsample = compute_decim_delay(b_stages, factors)
    '''

    n_pad_upsample = 0
    for i in reversed(range(len(factors))):
        if len(b_stages[i]) % 2 == 0:
            raise TypeError(
                'Decimation filters must be odd-order:'
                'stage %d has %d coefficients.' % i, len(b_stages[i]))
        n_pad_upsample = n_pad_upsample*factors[i] + len(b_stages[i]) - 1

    return n_pad_upsample


# pylint: disable=too-many-arguments, too-many-locals
def multi_decim(sig_in, b_stages, factors, z_in=0, sig_leftover=(),
                plot_start_time=None, discard_initial=True):
    '''
    Applies cascaded FIR filter and decimation stages to a signal.

    Example:

    Chunked data is handled as follows:

    ```python

    for i, chunk in enumerate(chunks):
        if i == 0:
            sig_out, z_out, sig_unused = multi_decim(
                chunk, b_stages, factors)
        else:
            chunk_out, z_out, sig_unused = multi_decim(
                chunk, b_stages, factors,
                z_in=z_out, sig_leftover=sig_unused, discard_initial=False)
            sig_out = np.hstack((sig_out, chunk_out))
    ```

    Use :func:`compute_decim_delay` to determine the number of extra samples
    to request in order to keep output samples aligned with input samples.

    :param sig_in: input samples
    :type sig_in: :class:`~numpy.array`
    :param b_stages: decimation coefficients
    :type b_stages: list of :class:`~numpy.array`
    :list factors: integer decimation factors for each stage
    :param z_in: filter delays
    :type z_in: list of :class:`~numpy.array` or scalar numeric

    :returns: output samples, filter delays and unused input samples
    :rtype: (:class:`~numpy.array`,
             list of :class:`~numpy.array`,
             :class:`~numpy.array`)
    '''
    assert len(b_stages) == len(factors)

    sig_in = np.hstack((sig_leftover, sig_in))

    total_decimation = np.prod(factors)

    if discard_initial:  # samples are consumed in loading of filters
        n_pad_upsample = compute_decim_delay(b_stages, factors)
        len_output = int((len(sig_in) - n_pad_upsample)/total_decimation)
        len_input_required = int(len_output*total_decimation) + n_pad_upsample
    else:
        len_output = int(len(sig_in)/total_decimation)
        len_input_required = int(len_output*total_decimation)

    if len_output < 1:  # not enough input samples; just pass signal through
        return np.array([]), z_in, sig_in

    # initialize signals and pass unused signal through to output
    sig_stages = [None]*(len(factors) + 1)
    sig_stages[0] = sig_in[:len_input_required]
    sig_unused = sig_in[len_input_required:]

    if isinstance(z_in, int) or isinstance(z_in, float):
        z_in = [z_in*lfilter_zi(b_stage, 1) for b_stage in b_stages]
    else:
        assert len(z_in) == len(factors)
        assert all([len(z_stage) + 1 == len(b_stage)
                    for z_stage, b_stage in zip(z_in, b_stages)])

    if discard_initial:
        first_indices = [len(b_stage) - 1 for b_stage in b_stages]
    else:
        first_indices = [0]*len(b_stages)

    # alternately filter and decimate according to stage specifications
    z_out = [None]*len(factors)
    for i, _ in enumerate(factors):
        sig_stages[i], z_out[i] = lfilter(b_stages[i], 1,
                                          sig_stages[i], zi=z_in[i])
        sig_stages[i + 1] = sig_stages[i][first_indices[i]::factors[i]]

    assert len(sig_stages[-1]) == len_output

    if plot_start_time is not None:  # plot result
        t_out = plot_start_time + get_times(
            sig_stages[-1], b_stages=b_stages, factors=factors,
            discard_initial=discard_initial)

        _, ax = plt.subplots()
        ax.plot(plot_start_time + get_times(sig_in), sig_in, label='input')
        ax.plot(t_out, sig_stages[-1], marker='x', label='output')
        ax.set_xlabel('Time [s]')
        ax.grid()
        ax.legend()

    return sig_stages[-1], z_out, sig_unused


def extract_decimation_coefficients(stages, verbose=False):
    '''
    Extract decimation factors and filter coefficients from a list of stages.
    '''

    b_stages = []
    factors = []
    for stage in stages:
        if (isinstance(stage, CoefficientsTypeResponseStage) and
                stage.decimation_factor > 1):
            if len(factors) == 0 and verbose:
                print('Input sample rate %g sps' %
                      stage.decimation_input_sample_rate)

            factors.append(stage.decimation_factor)
            b_stages.append(stage.numerator)
            if verbose:
                print('Filter with %d coefficients '
                      'and decimate by %d to %g sps'
                      % (len(stage.numerator), stage.decimation_factor,
                         stage.decimation_input_sample_rate /
                         stage.decimation_factor))
    if verbose:
        n_pad_upsample = compute_decim_delay(b_stages, factors)
        print('Filtering and decimation by %d consumes %d samples'
              % (np.prod(factors), n_pad_upsample))

    return b_stages, factors


def sample_hold_digitize(signal):
    '''
    Process signal as if sampled and held by a DAC, then digitized by an ADC.
    '''
    signal = np.concatenate((np.zeros((1,)), signal, np.zeros((1,))))
    signal = signal[:-1]/2 + signal[1:]/2
    return signal
