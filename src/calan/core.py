# -*- coding: utf-8 -*-
"""
A collection of utilities useful for station quality analysis.
"""

import os
import sys

import numpy as np
from scipy import fftpack
import scipy.signal as sp
import matplotlib.pyplot as plt

from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.client import FDSNException
from obspy.core.inventory import CoefficientsTypeResponseStage

from antelope_tools.utilities import DATETIME_FORMAT, pretty_duration
from calan.noise_survey_toolbox import (
    dataless2inventory, preferred_number)

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def minreal(lti_in, tolerance=0., f_norm=1, method='damping'):
    '''
    Removes (nearly) identical zero-pole pairs from a transfer function.

    The 'octave' method is that used in octave :func:`~control.minreal` and
    python-control :func:`~TransferFunction.minreal`.
    This method only considers the absolute distance between each
    pole `p` and zero `z`:

        `abs(z-p) < tolerance`

    The 'damping' method is an improvement on this method whereby
    pole-zero pairs must be farther apart in order to be considered
    cancelling when the damping is low.
    In this case poles and zeros are considered to cancel when the following
    condition is met:

        `abs(z-p)/sqrt(|z|*|p|)/sqrt(cos(angle(z))*cos(angle(p))) < tolerance`

    Parameters
    ----------
    lti_in: instance of :class:`~scipy.signal.lti`
        transfer function before cancellation
    tolerance: `float`, optional
        tolerance for cancellation
    f_norm: `float`, optional
        normalization frequency - anywhere mid-band
    method: `str`, optional
        'damping' or 'octave' as described above

    Returns
    -------
    lti_out: instance of :class:`~scipy.signal.lti`
        transfer function after cancellation
    '''

    assert isinstance(lti_in, sp.lti)
    tolerance = float(tolerance)
    assert tolerance >= 0
    f_norm = float(f_norm)
    assert f_norm >= 0
    assert method in ['damping', 'octave']

    z_mat = np.tile(lti_in.zeros, (len(lti_in.poles), 1)).transpose()
    p_mat = np.tile(lti_in.poles, (len(lti_in.zeros), 1))

    condition = np.abs(z_mat - p_mat)
    if method == 'damping':
        condition /= np.sqrt(np.abs(p_mat)*np.abs(z_mat)) / \
                     np.sqrt(np.cos(np.angle(p_mat)) * np.cos(np.angle(z_mat)))

        # deal with NaNs produced by 0/0
        condition[z_mat == p_mat] = 0

    # a zero may cancel only one pole and vice versa
    # thus only closest cancellation is retained
    cancel_indices = np.zeros(condition.shape, dtype=bool)
    while np.any(np.any(condition <= tolerance)):
        i, j = np.unravel_index(np.nanargmin(condition), condition.shape)
        condition[i, :] = np.inf
        condition[:, j] = np.inf
        cancel_indices[i, j] = True

    p_out = lti_in.poles[np.logical_not(np.any(cancel_indices, axis=0))]
    z_out = lti_in.zeros[np.logical_not(np.any(cancel_indices, axis=1))]

    k_out = float(abs(lti_in.freqresp(w=2*np.pi*f_norm)[1])) / \
        float(abs(sp.lti(z_out, p_out, 1).freqresp(w=2*np.pi*f_norm)[1]))

    return sp.lti(z_out, p_out, k_out)


def flip(ndarray, axis):
    '''
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.

    Borrowed from the future, numpy v1.12.dev0

    Parameters
    ----------
    m: array_like
        Input array.
    axis: integer
        Axis in array, which entries are reversed.

    Returns
    -------
    out: array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    Notes
    -----
    flip(m, 0) is equivalent to numpy.flipud(m).

    flip(m, 1) is equivalent to numpy.fliplr(m).
    '''
    if not hasattr(ndarray, 'ndim'):
        ndarray = np.asarray(ndarray)
    indexer = [slice(None)] * ndarray.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError('axis=%i is invalid for %i-dimensional input array'
                         % (axis, ndarray.ndim))
    return ndarray[tuple(indexer)]


def unwrap_mid(phase_in, f_in, f_midband=1, axis=-1):
    '''
    Unwraps phase data in the range starting at midband

    Unwrapping is done from -pi to pi starting at a specified midband
    frequency and working outwards.

    Parameters
    ----------
    phase_in: :class:`~numpy.array` or list
        phase data
    f_in: :class:`~numpy.array` or list
        frequencies corresponding to phases
    f_midband: float, optional
        midband frequency at which to start unwrapping

    Returns
    -------
    phase_out: :class:`~numpy.array`
        unwrapped phase data
    '''

    assert f_in.ndim == 1

    i_mid = np.argmin(np.abs(np.array(f_in)/f_midband - 1))
    phase_below = phase_in.take(np.arange(i_mid), axis)
    phase_above = phase_in.take(np.arange(i_mid, phase_in.shape[axis]), axis)

    phase_below = flip(np.unwrap(flip(phase_below, axis), axis), axis)
    phase_above = np.unwrap(phase_above, axis)

    return np.concatenate((phase_below, phase_above), axis)


def lti_from_zpsf(zeros, poles, sensitivity, frequency):
    '''
    Generate a LinearTimeInvariant model from zeros, poles, sensitivity and
    frequency at which sensitivity is specified.
    '''
    model = sp.lti(zeros, poles, 1)
    midband = abs(model.freqresp(2*np.pi*frequency)[1])
    return sp.lti(zeros, poles, sensitivity/midband)


def long_names(stream,
               parts=('network', 'station', 'location', 'channel')):
    '''Construct a list of names for the traces in a stream.'''
    return ['.'.join([trace.stats[part] for part in parts])
            for trace in stream]


def factor_names(stream,
                 parts=('network', 'station', 'location', 'channel')):
    '''
    Returns a tuple with the factored names for the traces in a stream.
    The first item in the tuple is the part which is common to all traces;
    the second item in the tuple is a list of the parts which differ.
    '''
    full_names = long_names(stream, parts=parts)
    sames = [letters[1:] == letters[:-1] for letters in zip(*full_names)]
    short_names = [''.join(letter for letter, same in zip(name, sames)
                           if not same) for name in full_names]
    common_name = ''.join(letter for letter, same in zip(full_names[0], sames)
                          if same)
    return common_name, short_names


def num_windows_welch(len_signal, len_fft, len_overlap=None):
    '''Number of windows resulting from Welch's method'''
    if len_overlap is None:
        len_overlap = int(len_fft/2)

    return int((len_signal - len_fft)/(len_fft - len_overlap)) + 1


def len_fft_welch(len_signal, num_windows=None, fraction_overlap=None):
    '''
    Recommended FFT length to achieve target number of windows using Welch's
    method
    '''
    if num_windows is None:
        num_windows = 30
    if fraction_overlap is None:
        fraction_overlap = 0.5

    return int(preferred_number(
        len_signal / ((num_windows - 0.5) * (1 - fraction_overlap) + 1)))


def window_times_welch(len_signal, len_fft, f_sample, len_overlap=None):
    '''
    Array of times of centers of windows resulting from Welch's method.

    Nearly verbatim from scipy.signal._spectral_helper().
    '''
    if len_overlap is None:
        len_overlap = int(len_fft/2)

    return np.arange(len_fft/2, len_signal - len_fft/2 + 1,
                     len_fft - len_overlap)/f_sample


def fft_frequencies(len_fft, f_sample, sides='onesided'):
    '''
    Array of frequencies expected from an FFT calculation.

    Nearly verbatim from scipy.signal._spectral_helper().
    '''

    len_fft = int(len_fft)
    if sides == 'twosided':
        num_freqs = len_fft
    elif sides == 'onesided':
        if len_fft % 2:
            num_freqs = int((len_fft + 1)/2)
        else:
            num_freqs = int(len_fft/2 + 1)

    frequencies = fftpack.fftfreq(len_fft, 1/f_sample)[:num_freqs]

    if sides != 'twosided' and not len_fft % 2:
        # get the last value correctly, it is negative otherwise
        frequencies[-1] *= -1

    return frequencies


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
                discard_initial=True):
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
        z_in = [z_in*sp.lfilter_zi(b_stage, 1) for b_stage in b_stages]
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
        sig_stages[i], z_out[i] = sp.lfilter(
            b_stages[i], 1, sig_stages[i], zi=z_in[i])
        sig_stages[i + 1] = sig_stages[i][first_indices[i]::factors[i]]

    assert len(sig_stages[-1]) == len_output

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


R_SERIES = {
    3: [1, 2, 5, 10],
    5: [1.00, 1.60, 2.50, 4.00, 6.30, 10.00],
    10: [1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.00],
    20: [1.00, 1.12, 1.25, 1.40, 1.60, 1.80, 2.00, 2.24, 2.50, 2.80,
         3.15, 3.55, 4.00, 4.50, 5.00, 5.60, 6.30, 7.10, 8.00, 9.00, 10.00],
    40: [1.00, 1.06, 1.12, 1.18, 1.25, 1.32, 1.40, 1.50, 1.60, 1.70,
         1.80, 1.90, 2.00, 2.12, 2.24, 2.36, 2.50, 2.65, 2.80, 3.00,
         3.15, 3.35, 3.55, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.30,
         5.60, 6.00, 6.30, 6.70, 7.10, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00],
    80: [1.00, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.22, 1.25, 1.28,
         1.32, 1.36, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75,
         1.80, 1.85, 1.90, 1.95, 2.00, 2.06, 2.12, 2.18, 2.24, 2.30,
         2.36, 2.43, 2.50, 2.58, 2.65, 2.72, 2.80, 2.90, 3.00, 3.07,
         3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.87, 4.00, 4.12,
         4.25, 4.37, 4.50, 4.62, 4.75, 4.87, 5.00, 5.15, 5.30, 5.45,
         5.60, 5.80, 6.00, 6.15, 6.30, 6.50, 6.70, 6.90, 7.10, 7.30,
         7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00],
}


E_SERIES = {
    6: [10, 15, 22, 33, 47, 68, 100],
    12: [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100],
    24: [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
         33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91, 100],
    48: [100, 105, 110, 115, 121, 127, 133, 140, 147, 154, 162, 169,
         178, 187, 196, 205, 215, 226, 237, 249, 261, 274, 287, 301,
         316, 332, 348, 365, 383, 402, 422, 442, 464, 487, 511, 536,
         562, 590, 619, 649, 681, 715, 750, 787, 825, 866, 909, 953, 1000],
    96: [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130,
         133, 137, 140, 143, 147, 150, 154, 158, 162, 165, 169, 174,
         178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232,
         237, 243, 249, 255, 261, 267, 274, 280, 287, 294, 301, 309,
         316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
         422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549,
         562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
         750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976, 1000],
    192: [100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 113, 114,
          115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132,
          133, 135, 137, 138, 140, 142, 143, 145, 147, 149, 150, 152,
          154, 156, 158, 160, 162, 164, 165, 167, 169, 172, 174, 176,
          178, 180, 182, 184, 187, 189, 191, 193, 196, 198, 200, 203,
          205, 208, 210, 213, 215, 218, 221, 223, 226, 229, 232, 234,
          237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 271,
          274, 277, 280, 284, 287, 291, 294, 298, 301, 305, 309, 312,
          316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 357, 361,
          365, 370, 374, 379, 383, 388, 392, 397, 402, 407, 412, 417,
          422, 427, 432, 437, 442, 448, 453, 459, 464, 470, 475, 481,
          487, 493, 499, 505, 511, 517, 523, 530, 536, 542, 549, 556,
          562, 569, 576, 583, 590, 597, 604, 612, 619, 626, 634, 642,
          649, 657, 665, 673, 681, 690, 698, 706, 715, 723, 732, 741,
          750, 759, 768, 777, 787, 796, 806, 816, 825, 835, 845, 856,
          866, 876, 887, 898, 909, 920, 931, 942, 953, 965, 976, 988, 1000]
}


def stdval(value, num=96, bump=0, preferred=None):
    """
    Computes nearest values in a standard-value series.

    For non-standard E-numbers, Renard numbers are used. Note that the
    standard E-series do not strictly follow the Renard number series,
    which is why lookup tables must be used. An optional variable "bump"
    specifies the number by which the series index is to be adjusted up or
    down, and is useful for ceiling/floor type operations.

    Arguments
    :param value: Values to which the nearest standard values are sought
    :param num: Number of preferred values in standard series,
        e.g. 96 for E96 series
    :type num: list[int]
    :param preferred: Preferred values, e.g. [1, 2, 5, 10]
    :type preferred::mod:numpy:array:
    :param float bump: Amount by which each series index is adjusted before
        choosing value

    Note that num=3 gives same result as preferred=[1, 2, 5, 10], but
    num=24 would not give the correct E24 series if it weren't overridden.

    :returns output: Nearest standard values after rounding
    """

    # we're going to need to do some elementwise operations
    x_type = type(value)
    value = np.asarray(value, dtype=float)

    # and some operations which depend on input being a column vector
    x_shape = value.shape
    value = np.reshape(value, (value.size, 1))

    # negative values will not be handled
    value[value < 0] = np.nan

    if preferred is None:
        if num in E_SERIES.keys():
            preferred = np.array(E_SERIES[num])
        elif num in R_SERIES.keys():
            preferred = np.array(R_SERIES[num])
        log_series = True
    else:
        # for the purpose of "bumping" it will be assumed that the preferred
        # values are approximately logarithmically-spaced and span a decade
        preferred = np.asarray(preferred, dtype=float)
        preferred = np.reshape(preferred, (1, preferred.size))
        num = preferred.size - 1
        log_series = False

    # bump input up or down as requested to support rounding up and down
    if log_series:
        value = value*10**(np.asarray(bump)/num)

    if preferred is not None:
        # determine how many digits of result to keep
        preferred = np.asarray(preferred, dtype=float)
        preferred = np.reshape(preferred, (1, preferred.size))
        digits = len('%d' % preferred[0][0])

        # compute multiplier for rounding
        multiplier = 10**np.floor(np.log10(value) - digits + 1)

        # shift input to have the right number of digits
        value = value/multiplier

        # find nearest standard value in a logarithmic sense
        pref_mat = np.tile(np.log10(preferred), (value.size, 1)).transpose()
        x_mat = np.tile(np.log10(value), (1, preferred.size)).transpose()

        log_dist = pref_mat - x_mat
        i_closest = np.argmin(np.abs(log_dist), 0)

        if log_series or bump == 0:
            output = preferred[0, i_closest]
        else:
            # for non-logarithmic series, bump just means round up or down
            if bump > 0 and log_dist[i_closest] < 0:
                if i_closest == preferred.size:
                    output = preferred[1]*10
                else:
                    output = preferred[i_closest + 1]

            elif bump < 0 and log_dist[i_closest] > 0:
                if i_closest == 1:
                    output = preferred[-1]/10
                else:
                    output = preferred[i_closest - 1]

        # restore correct number
        output = output[:, None]*multiplier

    else:
        # the nth power of a decade is the base
        base = 10**(1.0/num)

        # determine how many digits of result to keep
        digits = np.max((1, -np.round(np.log10(base - 1) - 1)))

        # compute the sequence number
        exponent = np.round(np.log(value)/np.log(base))

        # compute raw result
        raw = base**exponent

        # compute multiplier for rounding
        multiplier = 10**np.floor(np.log10(raw) - digits + 1)

        # round result to requested number of digits
        output = np.round(raw/multiplier)*multiplier

    output = np.reshape(output, x_shape)

    if (x_type is int) | (x_type is float):
        output = float(output)

    return output


def logspace(start, stop, num=12):
    """
    Computes logarithmically spaced vector of preferred numbers.

    See stdval.
    """

    log_start = np.floor(np.log10(start))
    log_stop = np.ceil(np.log10(stop))
    n_total = num*(log_stop-log_start) + 1
    temp = stdval(np.logspace(log_start, log_stop, num=n_total), num=num)
    return temp[np.bitwise_and(temp >= start, temp <= stop)]


def truncnorm_shape(mean, std, clip_a, clip_b=None):
    '''
    Convert mean, standard deviation and clip levels to
    :class:`~scipy.stats.truncnorm' shape parameters.

    :returns: a, b
    '''
    if clip_b is None:
        clip_b = - clip_a
    shape_a, shape_b = (clip_a - mean) / std, (clip_b - mean) / std

    return shape_a, shape_b


def subplots_squeeze(fig, hspace=None, wspace=None):
    '''
    For now this just supports the case of multiple axes stacked vertically,
    removing space between them and removing tick labels which would overlap.
    '''

    axes_indices = [[child.colNum, child.rowNum]
                    for child in fig.get_children()[1:]]
    num_cols, num_rows = np.max(axes_indices, axis=0) + 1
    axes = np.reshape(fig.axes, (num_rows, num_cols))

    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if hspace < 0.05 and num_rows > 1:
        for i, ax in enumerate(axes[:, 0]):
            if i > 0:
                ax.yaxis.get_major_ticks()[-1].label.set_visible(False)
            if i < num_rows - 1:
                ax.yaxis.get_major_ticks()[0].label.set_visible(False)


class Stft():
    '''
    Short-term fourier auto- and cross-spectra between input and output.
    '''
    def __init__(self, f=None, t=None, p_xx=None, p_yy=None, p_xy=None):

        self.f = f
        self.t = t
        self.p_xx = p_xx
        self.p_yy = p_yy
        self.p_xy = p_xy

    def compute(self, x, y, f_sample, len_fft, len_overlap):
        '''
        Each segment is detrended by removing a constant value before
        application of a 'hanning' window.
        '''

        # pylint: disable=protected-access
        self.f, self.t, self.p_xy = sp.spectral._spectral_helper(
            x, y,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')

        self.p_xx = sp.spectral._spectral_helper(
            x, x,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]

        self.p_yy = sp.spectral._spectral_helper(
            y, y,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]
        # pylint: enable=protected-access


class StreamAnalyzer():
    '''
    Base class for a :class:`~obspy.Stream`-based signal analyzer.
    '''

    def __init__(self, fdsn_server='http://132.156.41.208:6062',
                 cache_format='MSEED', verbose=True):
        '''
        Sets up FDSN server for later use.
        '''
        if fdsn_server is None or fdsn_server == '':
            self.client = None
            print('You are working offline.')
        else:
            self.client = Client(fdsn_server)
            print('FDSN server: \n\t%s' % self.client.base_url)

        self.cache_format = cache_format
        self.verbose = verbose
        self.stream = None

    def make_cache_name(self):
        '''
        Construct a name for cached copy of a stream or an inventory.
        '''
        start = np.max([trace.stats.starttime for trace in self.stream])
        end = np.min([trace.stats.endtime for trace in self.stream])
        start_string = start.strftime(DATETIME_FORMAT)
        start_string = start_string.replace(':', '-').replace(' ', '_')
        duration_string = pretty_duration(end - start)
        common_name = factor_names(self.stream)[0]

        return '_'.join([common_name, start_string, duration_string])

    def load_stream(self, start, end,
                    input_file=None, inventory_dataless=None, networks=None,
                    stations=None, locations=None, channels=None):
        '''
        Load stream and attach inventory from files or FDSN server.
        '''
        start = UTCDateTime(start)
        end = UTCDateTime(end)
        if isinstance(networks, str):
            networks = [networks]
        if isinstance(stations, str):
            stations = [stations]
        if isinstance(locations, str):
            locations = [locations]
        if isinstance(channels, str):
            channels = [channels]

        if input_file is None:
            if self.verbose:
                print('Requesting data from FDSN server: \n\t%s'
                      % self.client.base_url)
            try:
                self.stream = self.client.get_waveforms(
                    network=','.join(networks), station=','.join(stations),
                    location=','.join(locations), channel=','.join(channels),
                    starttime=start, endtime=end)
            except FDSNException as ex:
                print('Networks: ', ','.join(networks))
                print('Stations: ', ','.join(stations))
                print('Locations:', ','.join(locations))
                print('Channels: ', ','.join(channels))
                print('Start:    ', str(start))
                print('End:    ', str(end))
                raise ex

            input_file = '.'.join([self.make_cache_name(), self.cache_format])
            if self.verbose:
                print('Caching %s locally as: \n\t%s' % (self.cache_format,
                                                         input_file))
            self.stream.write(input_file, format=self.cache_format)
        else:
            if self.verbose:
                print('Reading data from %s file: \n\t%s' % (self.cache_format,
                                                             input_file))
            self.stream = read(input_file)

        if inventory_dataless is None:
            if self.verbose:
                print('Requesting responses from FDSN server: \n\t%s'
                      % self.client.base_url)
            inventory = self.client.get_stations(
                network=','.join(networks), station=','.join(stations),
                location=','.join(locations), channel=','.join(channels),
                startbefore=start, endafter=end,
                level='response', includerestricted=True)

            inventory_dataless = self.make_cache_name() + '.xml'
            if self.verbose:
                print('Caching StationXML locally as: \n\t%s'
                      % inventory_dataless)
            inventory.write(inventory_dataless, format='STATIONXML')
        else:
            if self.verbose:
                print('Reading responses from StationXML file: \n\t%s'
                      % inventory_dataless)
            inventory = dataless2inventory(inventory_dataless)

        self.stream = self.stream.merge().split().sort()
        self.stream.attach_response(inventory)

    def f_sample(self):
        '''Return stream sampling rate'''
        return self.stream[0].stats.sampling_rate

    @staticmethod
    def _mean(p_xy):
        '''Finishing touch on Welch's method.'''

        if len(p_xy.shape) >= 2 and p_xy.size > 0:
            if p_xy.shape[-1] > 1:
                p_xy = p_xy.mean(axis=-1)
            else:
                p_xy = np.reshape(p_xy, p_xy.shape[:-1])
        return p_xy

    def save_image(self, fig=None, option_list=None):
        '''
        Save a figure with an automatically descriptive file name.
        options: comma-separated string or list of strings, optional
            'system' or 'cal' divides out that part of the nominal response
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
            file_parts += [option for option in option_list if len(option) > 0]

        if self.stream is not None:
            if hasattr(self, 'info'):
                start_string = str(self.info['start'].date)
            else:
                start = np.max([trace.stats.starttime
                                for trace in self.stream])
                start_string = start.strftime(DATETIME_FORMAT)
            common_name = factor_names(self.stream)[0]

            file_parts += [common_name, start_string]
        else:
            if hasattr(self, 'file'):
                calibration_name = self.info['file'].replace('.gz', '')
                calibration_name = calibration_name.replace('.wav', '')
                file_parts += [calibration_name]

        file_name = '_'.join(file_parts) + '.png'

        if self.verbose:
            print('Saving to', file_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    def plot_stream(self, save=False):
        '''
        Quick plot of active part of calibration
        '''
        self.stream.plot(handle=True)

        if save:
            self.save_image()
