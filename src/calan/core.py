# -*- coding: utf-8 -*-
"""
A collection of utilities useful for station quality analysis.
"""
# pylint: disable=logging-not-lazy
# for Python 2 & 3 compatibility
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import queue
import inspect
from glob import glob
from time import time
from tempfile import gettempdir

import requests
import numpy as np
import pandas as pd
from scipy import fftpack
import scipy.signal as sp

from obspy import read_inventory, UTCDateTime
from obspy.clients import fdsn
from obspy.core.inventory import CoefficientsTypeResponseStage

from catalogue_tools.utilities import get_logger, string_list, preferred_number

from calan import chis_archive

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PACKAGE = os.path.basename(os.path.dirname(__file__))

DEFAULT_FDSN_SERVERS = (
    'http://fdsn.seismo.nrcan.gc.ca',  # production, SeisComP3
    'http://sc3-stage.seismo.nrcan.gc.ca',  # staging, seisComP3
    'IRIS',
    )


def get_clients(servers=None, test_timeout=2):
    '''
    Given a list of URLs, returns a list of FDSN clients.
    '''
    if servers is None:
        servers = [chis_archive.DEFAULT_ROOT] + list(DEFAULT_FDSN_SERVERS)
    servers = string_list(servers)

    clients = []
    logger = get_logger(__name__)
    if servers is not None:
        for server in servers:
            fdsn_server = (server.startswith('http') or
                           server in fdsn.URL_MAPPINGS)

            if fdsn_server:
                if server.startswith('http'):
                    test_server = server
                else:
                    test_server = fdsn.URL_MAPPINGS[server]

                try:
                    requests.get(test_server, timeout=test_timeout)
                except (requests.Timeout, requests.ConnectionError,
                        queue.Empty) as ex:
                    logger.debug(repr(ex))
                    continue
                except requests.TooManyRedirects as ex:
                    logger.debug(repr(ex))
            else:
                if not os.path.isdir(server):
                    logger.warning(
                        'CHIS archive %s not available. Ignoring.' % server)
                    continue

            try:
                if fdsn_server:
                    clients.append(fdsn.Client(server))
                else:
                    clients.append(chis_archive.Client(server))
            except fdsn.client.FDSNException as ex:
                logger.warning(repr(ex))

    return clients


def fdsn_error_message(ex):
    '''
    Cleans up certain obspy.clients.fdsn exception messages.
    '''
    lines = ex.args[0].split('\n')
    if 'No data available' in lines[0]:
        msg = lines[0]
    else:
        msg = ' '.join(lines)
    return msg


STATIONXML_CONVERTER_FILE = 'stationxml-converter-1.0.9.jar'
STATIONXML_CONVERTER = next(iter(glob(
    os.path.join(ROOT, '**', STATIONXML_CONVERTER_FILE))), None)
if STATIONXML_CONVERTER is None:
    print('WARNING: StationXML converter "%s" not found. '
          'Cannot convert dataless2inventory ' % STATIONXML_CONVERTER_FILE)


def dataless2inventory(inventory_dataless, inventory_source='GSC'):
    '''
    Convert a dataless SEED to an :class:`~obspy.core.inventory.Inventory`.
    A StationXML file is produced in the same directory as the input file.
    '''
    inventory_xml = dataless2stationxml(inventory_dataless,
                                        inventory_source=inventory_source)
    return read_inventory(inventory_xml)


def dataless2stationxml(inventory_dataless, inventory_source='GSC'):
    '''
    Convert a dataless SEED to an :class:`~obspy.core.inventory.Inventory`.
    A StationXML file is produced in the same directory as the input file.
    '''
    logger = get_logger(__name__)
    if not os.path.isfile(inventory_dataless):
        logger.warning('Dataless SEED file "%s" not found'
                       % inventory_dataless)

    inventory_xml = inventory_dataless.replace('.dataless', '.xml')

    if not os.path.isfile(inventory_xml):
        os.system(
            'java -jar %s --xml --prettyprint --source %s --output %s %s' %
            (STATIONXML_CONVERTER, inventory_source, inventory_xml,
             inventory_dataless))

    return inventory_xml


def inventory2dataless(inventory_xml):
    '''
    Convert a dataless SEED to an :class:`~obspy.core.inventory.Inventory`.
    A StationXML file is produced in the same directory as the input file.
    '''
    logger = get_logger(__name__)
    if not os.path.isfile(inventory_xml):
        logger.warning('StationXML file "%s" not found'
                       % inventory_xml)

    inventory_dataless = inventory_xml.replace('.xml', '.dataless')

    if not os.path.isfile(inventory_dataless):
        os.system(
            'java -jar %s --seed --output %s %s' %
            (STATIONXML_CONVERTER, inventory_dataless, inventory_xml))

    return inventory_dataless


def _elapsed_since(tick):
    return str(pd.to_timedelta(round(time() - tick), 's')).split()[-1]


def get_chis_stations(level='response', minlatitude=35, maxlatitude=90,
                      maxlongitude=-40, minlongitude=-170):
    '''
    Returns an inventory of all stations for which CHIS has waveform data.

    If a cache is found, it is used, for speedup.

    Example
    -------
    inventory = get_chis_stations()
    INFO     Client: http://192.168.41.158:8080
    INFO     Read GSC inventory via SeisComP3: 00:01:03
    INFO     Cached /tmp/response_minlatitude35_maxlatitude90.xml: 00:00:09

    inventory = get_chis_stations()
    INFO     Cache: /tmp/response_minlatitude35_maxlatitude90.xml
    INFO     Elapsed: 00:00:17
    '''
    logger = get_logger(__name__)

    args, _, _, defaults = inspect.getfullargspec(get_chis_stations)[:4]
    inventory_file = os.path.join(
        gettempdir(),
        '_'.join('%s%s' % (arg, default)
                 for arg, default in zip(args, defaults)) + '.xml')
    inventory_file = inventory_file.replace('level', '')

    if os.path.isfile(inventory_file):
        logger.info('Cache: ' + inventory_file)
        tick = time()
        inventory = read_inventory(inventory_file)
        logger.info('Elapsed: ' + _elapsed_since(tick))
    else:
        try:
            chis_fdsn_client = fdsn.Client(DEFAULT_FDSN_SERVERS[0])
        except fdsn.client.FDSNException:
            chis_fdsn_client = fdsn.Client(DEFAULT_FDSN_SERVERS[2])

        logger.info('Client: ' + chis_fdsn_client.base_url)

        tick = time()
        inventory = chis_fdsn_client.get_stations(
            level=level, minlatitude=minlatitude, maxlatitude=maxlatitude,
            maxlongitude=maxlongitude, minlongitude=minlongitude)
        logger.info('Read %s inventory via %s: %s' %
                    (inventory.sender.upper(), inventory.source,
                     _elapsed_since(tick)))

        tick = time()
        inventory.write(inventory_file, format='StationXML')
        logger.info('Cached %s: %s' % (inventory_file, _elapsed_since(tick)))


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
        with np.errstate(divide='ignore', invalid='ignore'):
            condition /= np.sqrt(np.abs(p_mat)*np.abs(z_mat)) / \
                         np.sqrt(np.cos(np.angle(p_mat)) *
                                 np.cos(np.angle(z_mat)))

        # deal with NaNs produced by 0/0
        condition[z_mat == p_mat] = 0

    # a zero may cancel only one pole and vice versa
    # thus only closest cancellation is retained
    cancel_indices = np.zeros(condition.shape, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
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
               parts=('network', 'station', 'location', 'channel'),
               widths=(2, 5, 2, 3)):
    '''Construct a list of names for the traces in a stream.'''
    return ['.'.join([('%' + str(width) + 's') % trace.stats[part]
                      for part, width in zip(parts, widths)])
            for trace in stream]


def factor_names(stream):
    '''
    Returns a tuple with the factored names for the traces in a stream.
    The first item in the tuple is the part which is common to all traces;
    the second item in the tuple is a list of the parts which differ.
    '''
    full_names = long_names(stream)
    sames = [letters[1:] == letters[:-1] for letters in zip(*full_names)]
    short_names = [''.join(letter for letter, same in zip(name, sames)
                           if not same) for name in full_names]
    common_name = ''.join(letter for letter, same in zip(full_names[0], sames)
                          if same)
    common_name = '.'.join(part.strip() for part in common_name.split('.'))
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

    if isinstance(z_in, (int, float)):
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


def extract_decimation_coefficients(stages):
    '''
    Extract decimation factors and filter coefficients from a list of stages.
    '''

    logger = get_logger(__name__)
    b_stages = []
    factors = []
    for stage in stages:
        if (isinstance(stage, CoefficientsTypeResponseStage) and
                stage.decimation_factor > 1):
            if not factors:
                logger.info('Input sample rate %g sps' %
                            stage.decimation_input_sample_rate)

            factors.append(stage.decimation_factor)
            b_stages.append(stage.numerator)
            logger.info(
                'Filter with %d coefficients and decimate by %d to %g sps'
                % (len(stage.numerator), stage.decimation_factor,
                   stage.decimation_input_sample_rate/stage.decimation_factor))
    n_pad_upsample = compute_decim_delay(b_stages, factors)
    logger.info('Filtering and decimation by %d consumes %d samples'
                % (np.prod(factors), n_pad_upsample))

    return b_stages, factors


def truncnorm_shape(mean, std, clip_b, clip_a=None):
    '''
    Convert mean, standard deviation and clip levels to
    :class:`~scipy.stats.truncnorm' shape parameters.

    :returns: a, b
    '''
    if clip_a is None:
        clip_a = -clip_b
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
    Short-term fourier auto- and cross-spectra between input (x) and output
    (y) signals.
    '''
    def __init__(self, f=None, t=None, p_xx=None, p_yy=None, p_xy=None):

        self.f = f
        self.t = t
        self.p_xx = p_xx
        self.p_yy = p_yy
        self.p_xy = p_xy

    @staticmethod
    def _mean(p_xy):
        '''
        Finishing touch of Welch's method when deriving results.
        '''
        if len(p_xy.shape) >= 2 and p_xy.size > 0:
            if p_xy.shape[-1] > 1:
                p_xy = p_xy.mean(axis=-1)
            else:
                p_xy = np.reshape(p_xy, p_xy.shape[:-1])
        return p_xy

    def compute(self, x, y, f_sample, len_fft, len_overlap, window='hann'):
        '''
        Each segment is detrended by removing a constant value before
        application of a window.
        '''
        logger = get_logger(self.__class__.__name__ + ':' + __name__)
        f_expected = fft_frequencies(len_fft, f_sample)
        num_samples = x.shape[0]
        num_windows = num_windows_welch(num_samples, len_fft, len_overlap)
        logger.info(
            '%d segments from %g to %g Hz' %
            (num_windows, f_expected[1], f_expected[-1]))

        # pylint: disable=protected-access
        self.f, self.t, self.p_xy = sp.spectral._spectral_helper(
            x, y, window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')

        self.p_xx = sp.spectral._spectral_helper(
            x, x, window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]

        self.p_yy = sp.spectral._spectral_helper(
            y, y,window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]
        # pylint: enable=protected-access

    def trim(self, low_frequency_points=5, high_frequency_fraction=0.8):
        '''
        Trim low- and high-frequency points.

        Typically low-frequency measurements are spoiled by imperfect DC
        removal. Similarly high-frequency measurements beyond the decimation
        filter corner are not useful.
        '''
        keep = ((self.f >= self.f[low_frequency_points]) &
                (self.f < self.f[-1]*high_frequency_fraction))
        self.f = self.f[keep]
        self.p_xx = self.p_xx[..., keep, :]
        self.p_yy = self.p_yy[..., keep, :]
        self.p_xy = self.p_xy[..., keep, :]

    def get_transfer_function(self, alpha=0):
        '''
        Return transfer function estimate (from input, x, to output, y),
        differentiated alpha times.
        '''
        return (self._mean(self.p_xy) /
                self._mean(self.p_xx))*(1j*2*np.pi*self.f)**alpha

    def get_coherence_squared(self):
        '''
        Return squared coherence.
        '''
        return (np.abs(self._mean(self.p_xy))**2 /
                (self._mean(self.p_xx)*self._mean(self.p_yy)))


def _missing_samples(delta, sampling_rate):
    return np.rint(np.fabs(delta)*sampling_rate)


def is_complete(stream, trace_ids=(), start=pd.Timestamp(0),
                end=pd.Timestamp.now(), tolerance=0.5):
    '''
    Lightweight test whether stream is complete.
    '''
    trace_ids = string_list(trace_ids)
    if trace_ids is None:
        trace_ids = sorted(list(set(trace.id for trace in stream)))

    stream = stream.sort()
    for trace_id in trace_ids:
        traces = stream.select(id=trace_id).traces

        # ensure that there is some data
        if not traces:
            return False

        # check start
        if (traces[0].stats.starttime > UTCDateTime(start) +
                tolerance/traces[0].stats.sampling_rate):
            return False

        # check end
        if (traces[-1].stats.endtime < UTCDateTime(end) -
                tolerance/traces[-1].stats.sampling_rate):
            return False

        for i in range(len(traces) - 1):
            # check that sample rate doesn't change
            if traces[i].stats.delta != traces[i + 1].stats.delta:
                return False

            # compute gap (positive) or overlap (negative)
            delta = (traces[i + 1].stats['starttime'].timestamp -
                     traces[i].stats['endtime'].timestamp +
                     traces[i].stats.delta)

            # check that any overlap is not larger than trace coverage
            if delta < 0:
                temp = (traces[i + 1].stats['endtime'].timestamp -
                        traces[i + 1].stats['starttime'].timestamp)
                if (delta * -1) > temp:
                    delta = -1 * temp

            missing_samples = _missing_samples(
                delta, traces[i].stats['sampling_rate'])
            if missing_samples > 0:
                return False

    return True


ID_COLUMNS = ['network', 'station', 'location', 'channel']
GAP_COLUMNS = ['starttime', 'endtime', 'duration', 'samples']


def gap_list(stream, trace_ids=(), start=pd.Timestamp(0),
             end=pd.Timestamp.now(), tolerance=0.5):
    '''
    Construct a dataframe of gaps, including start & end gaps.

    If an empty stream is provided the returned value are empty dataframes
    with the correct columns.

    Returns
    -------
    2-tuple of pd.DataFrame:
        gaps_df, overlap_df
    '''
    if trace_ids is None:
        trace_ids = ()
    if not isinstance(start, pd.Timestamp):
        start = pd.to_datetime(start.datetime)
    if not isinstance(end, pd.Timestamp):
        end = pd.to_datetime(end.datetime)

    if stream:
        gaps_df = pd.DataFrame(stream.get_gaps(),
                               columns=ID_COLUMNS + GAP_COLUMNS)

        gaps_df.insert(
            0, 'id', ['.'.join(items)
                      for _, items in gaps_df[ID_COLUMNS].iterrows()])
        gaps_df['sampling_rate'] = np.round(gaps_df.samples/gaps_df.duration)
        gaps_df.starttime = gaps_df.starttime.apply(
            lambda item: pd.to_datetime(item.datetime))
        gaps_df.endtime = gaps_df.endtime.apply(
            lambda item: pd.to_datetime(item.datetime))
    else:
        gaps_df = pd.DataFrame(
            columns=['id'] + ID_COLUMNS + GAP_COLUMNS + ['sampling_rate'])

    if not trace_ids:
        trace_ids = sorted(set(trace.id for trace in stream))
    if not trace_ids:
        raise RuntimeError('Empty streams require trace_ids be specified.')

    for trace_id in trace_ids:
        id_stream = stream.select(id=trace_id)
        if not id_stream:
            duration = (end - start).total_seconds()
            series = pd.Series({
                'id': trace_id,
                'network': trace_id.split('.')[0],
                'station': trace_id.split('.')[1],
                'location': trace_id.split('.')[2],
                'channel': trace_id.split('.')[3],
                'starttime': start,
                'endtime': end,
                'duration': duration,
                'samples': -1,
                'sampling_rate': np.NaN,
                })
            gaps_df = gaps_df.append(series, ignore_index=True)

    for trace_id in trace_ids:
        id_stream = stream.select(id=trace_id)
        if not id_stream:
            continue
        id_stream.sort(keys=['starttime'])
        trace = id_stream[0]
        sampling_rate = trace.stats.sampling_rate
        tol = pd.to_timedelta(tolerance/sampling_rate, 's')
        trace_start = pd.to_datetime(trace.stats.starttime.datetime)
        if trace_start > start + tol:
            duration = (trace_start - start).total_seconds()
            series = pd.Series({
                'id': trace_id,
                'network': trace.stats.network,
                'station': trace.stats.station,
                'location': trace.stats.location,
                'channel': trace.stats.channel,
                'starttime': start,
                'endtime': trace_start,
                'duration': duration,
                'samples': _missing_samples(duration, sampling_rate),
                'sampling_rate': sampling_rate,
                })
            gaps_df = gaps_df.append(series, ignore_index=True)

    for trace_id in trace_ids:
        id_stream = stream.select(id=trace_id)
        if not id_stream:
            continue
        id_stream.sort(keys=['endtime'])
        trace = id_stream[-1]
        sampling_rate = trace.stats.sampling_rate
        tol = pd.to_timedelta(tolerance/sampling_rate, 's')
        trace_end = pd.to_datetime(trace.stats.endtime.datetime)
        if trace_end < end - tol:
            duration = (end - trace_end).total_seconds()
            series = pd.Series({
                'id': trace_id,
                'network': trace.stats.network,
                'station': trace.stats.station,
                'location': trace.stats.location,
                'channel': trace.stats.channel,
                'starttime': trace_end,
                'endtime': end,
                'duration': duration,
                'samples': _missing_samples(duration, sampling_rate),
                'sampling_rate': sampling_rate,
                })
            gaps_df = gaps_df.append(series, ignore_index=True)

        gaps_df.sort_values(by=['starttime', 'endtime'],
                            ascending=[True, False], inplace=True)
        gaps_df.reset_index(inplace=True, drop=True)

    return gaps_df[gaps_df.duration > 0], gaps_df[gaps_df.duration <= 0]


def fraction_available(trace_ids, start, end, gaps_df):
    '''
    Compute fraction of requested data which is available.
    '''
    trace_ids = string_list(trace_ids)

    expected_duration = len(trace_ids)*((UTCDateTime(end) -
                                         UTCDateTime(start)))
    gap_duration = gaps_df.loc[gaps_df.id.isin(trace_ids)].duration.sum()
    if not expected_duration:
        return np.NaN

    return 1 - gap_duration/expected_duration


def log_availability(logger, gaps_df, trace_ids, start, end,
                     column='duration'):
    '''
    Given a gap listing, summarize availability to a log file.
    '''
    gaps_df = gaps_df.copy()
    num_gaps = gaps_df.shape[0]
    daylong = np.abs((UTCDateTime(end) - UTCDateTime(start)) - 24*60*60) < 3600

    if num_gaps == 0:
        return

    assert column in ['duration', 'samples']
    if column == 'duration':
        gap_unit = 's'
    else:
        gap_unit = 'sample'

    common_id = ''.join(chars[0] for chars in zip(*trace_ids)
                        if len(set(chars)) == 1).strip('.')

    percent_available = 100*fraction_available(trace_ids, start, end, gaps_df)
    logger.info(
        '%s was %.1f%% complete with %d gap(s), e.g.:' %
        (common_id, percent_available, num_gaps))

    gaps_df.loc[:, 'note'] = ''

    # identify largest first and last gaps
    gaps_df.sort_values(by=['endtime', 'starttime'],
                        ascending=[False, True], inplace=True)
    last = gaps_df.index[0]
    gaps_df.sort_values(by=['starttime', 'endtime'],
                        ascending=[True, False], inplace=True)
    first = gaps_df.index[0]
    if first == last:
        gaps_df.at[first, 'note'] = 'first,last'
    else:
        gaps_df.at[first, 'note'] = 'first'
        gaps_df.at[last, 'note'] = 'last'

    # compute gap statistics
    percentiles = [0.05, 0.5, 0.95]
    keys = ['mode', '50%', 'max', 'min', '95%', '5%']
    notes = ['mode', 'median', 'largest', 'smallest', '95th', '5th']
    stats = gaps_df[column].describe(percentiles=percentiles)
    stats['mode'] = gaps_df[column].mode()[0]

    # eliminate redundant statistics
    for key in keys:
        if key in stats:
            stats = stats[(stats != stats[key]).values |
                          (stats.index == key)]

    # label gaps
    for key, note in zip(keys, notes):
        if key in stats:
            indices = gaps_df[column] == stats[key]
            gaps_df.loc[indices, 'note'] = \
                [','.join([item, note]) if item else note
                 for item in gaps_df.loc[indices, 'note']]

    # remove redundancies
    gaps_df = gaps_df[gaps_df.note != '']
    if gaps_df.at[first, 'note'] != 'first':
        gaps_df = gaps_df[gaps_df.note !=
                          gaps_df.at[first,
                                     'note'].replace('first,', '')]
    if gaps_df.at[last, 'note'] != 'last':
        gaps_df = gaps_df[gaps_df.note !=
                          gaps_df.at[last,
                                     'note'].replace('last,', '')]
    gaps_df = gaps_df.drop_duplicates(subset='note')

    for _, gap in gaps_df.iterrows():
        if daylong:
            logger.info(
                '%s: %s start of %.3g %s gap (%s)'
                % (gap.id, str(gap.starttime.time())[:-3],
                   gap[column], gap_unit, gap.note))
        else:
            logger.info(
                '%s: %s start of %.3g %s gap (%s)'
                % (gap.id, str(gap.starttime)[:-3],
                   gap[column], gap_unit, gap.note))
