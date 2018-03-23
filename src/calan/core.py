# -*- coding: utf-8 -*-
"""
A collection of utilities useful for station quality analysis.
"""
# for Python 2 & 3 compatibility
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
import warnings
from glob import glob

import numpy as np
from scipy import fftpack
import scipy.signal as sp
import matplotlib.pyplot as plt

from obspy import read, read_inventory, UTCDateTime
from obspy.clients.fdsn.client import FDSNException
from obspy.core.inventory import CoefficientsTypeResponseStage
from obspy.core.util.attribdict import AttribDict

from catalogue_tools.core import short_utc
from catalogue_tools.utilities import (
    get_logger, string_list, pretty_duration, preferred_number,
    fdsn_error_message)
from catalogue_tools.css2seed import GscStationInfo

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(FILE_NAME)[0] + '.log'

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
    logger = get_logger(__name__)
    if not os.path.isfile(inventory_dataless):
        logger.warning('Dataless SEED file "%s" not found'
                       % inventory_dataless)

    inventory_xml = inventory_dataless.replace('.dataless', '.xml')

    if not os.path.isfile(inventory_xml):
        os.system('java -jar %s --xml --source %s --output %s %s' % (
            STATIONXML_CONVERTER, inventory_source, inventory_xml,
            inventory_dataless))

    return read_inventory(inventory_xml)


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
            if len(factors) == 0:
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


class StreamAnalyzer(GscStationInfo):
    '''
    Base class for a :class:`~obspy.Stream`-based signal analyzer.
    '''
    CACHE = 'cache'

    def __init__(self, fdsn_servers=None, clients=None, cache_format='MSEED',
                 log_level='INFO', log_file_name=LOG_FILE_NAME):
        '''
        Sets up FDSN server for later use.
        '''
        super(StreamAnalyzer, self).__init__(
            fdsn_servers=fdsn_servers, clients=clients,
            log_level=log_level, log_file_name=log_file_name)

        if len(self.clients) == 0:
            self.logger.info('You are working offline.')

        self.cache_format = cache_format
        self.stream = None
        self.name = ''

    def __str__(self):
        lines = ['Clients:']
        lines += ['\t' + str(client).split('\n')[0] for client in self.clients]
        if self.stream:
            lines += self.stream.__str__(extended=True).split('\n')
        return '\n'.join(lines)

    def make_cache_name(self):
        '''
        Construct a name for cached copy of a stream or an inventory.
        '''
        if self.stream is None:
            return None

        start = np.max([trace.stats.starttime for trace in self.stream])
        end = np.min([trace.stats.endtime for trace in self.stream])
        start_string = short_utc(start)
        start_string = start_string.replace(':', '-').replace(' ', '_')
        duration_string = pretty_duration(end - start)
        common_name = factor_names(self.stream)[0]
        common_name = common_name.replace(' ', '').replace('..', '.')
        common_name = common_name.replace('..', '.')
        if common_name.startswith('.'):
            common_name = common_name[1:]

        return os.path.join(
            self.CACHE, '_'.join([common_name, start_string, duration_string]))

    def load_stream(self, start, end,
                    input_file=None, inventory_dataless=None, networks=None,
                    stations=None, locations=None, channels=None,
                    minimum_sampling_rate_sps=10,
                    response=True, cache=True, fill_value=None):
        '''
        Load stream and attach inventory from files or FDSN clients. Files, if
        specified, are loaded first, then self.clients are searched, in order,
        for any remaining combinations of `networks`, `stations`, `locations`
        and `channels`, until an instance of each `station` in `stations` is
        found.

        Note
        ----
        The networks, channels and locations arguments only serve to narrow the
        scope of the search for the specified stations. Thus, There is no way
        to use this method to return only ``N1.STN1`` and ``N2.STN2`` if
        ``N1.STN2`` or ``N2.STN1`` exist; in that case the result a request for
        ``stations=['STN1', 'STN2']`` and ``networks=['N1', 'N2]`` must
        subsequently be narrowed using
        :func:`~obspy.core.stream.Stream.select`.
        '''
        start = UTCDateTime(start)
        end = UTCDateTime(end)
        if networks is None:
            networks = '*'
        if stations is None:
            stations = '*'
        if locations is None:
            locations = '*'
        if channels is None:
            channels = '*'
        networks = string_list(networks)
        stations = string_list(stations)
        locations = string_list(locations)
        channels = string_list(channels)

        if input_file is not None:
            self.logger.info('Reading data from %s file: %s'
                             % (self.cache_format, input_file))
            self.stream = read(input_file)
        else:
            self.stream = None

        for client in self.clients:
            if 'dataselect' not in client.services:
                continue

            if self.stream is None:
                remaining = stations
            else:
                remaining = [station for station in stations
                             if station not in set([trace.stats.station
                                                    for trace in self.stream])]
            if len(remaining) == 0:
                break

            if hasattr(client, 'base_url'):
                self.logger.info('Trying URL: ' + client.base_url)
            elif hasattr(client, 'sds_root'):
                self.logger.info('Trying filesystem: ' + client.sds_root)
            else:
                continue

            self.logger.info('Requesting stations: ' + ', '.join(remaining))
            try:
                partial_stream = client.get_waveforms(
                    network=','.join(networks),
                    station=','.join(remaining),
                    location=','.join(locations),
                    channel=','.join(channels),
                    starttime=start - 1, endtime=end + 1)

            except FDSNException as ex:
                partial_stream = None
                self.logger.warning(fdsn_error_message(ex))

            if partial_stream is not None:
                partial_stream.traces = [
                    trace for trace in partial_stream.traces
                    if trace.stats.sampling_rate >= minimum_sampling_rate_sps]
                partial_stream.merge()

                partial_stream.sort(keys=['starttime'])
                if partial_stream[-1].stats.starttime > start:
                    self.logger.warning(
                        '%s starts at %s, after requested %s'
                        % (partial_stream[0].id,
                           partial_stream[-1].stats.starttime, start))

                partial_stream.sort(keys=['endtime'])
                if partial_stream[0].stats.endtime < end:
                    self.logger.warning(
                        '%s ends at %s, before requested %s'
                        % (partial_stream[0].id,
                           partial_stream[0].stats.endtime, end))

                partial_stream.trim(starttime=start, endtime=end)
                try:
                    partial_stream = partial_stream.split()
                except ZeroDivisionError as ex:
                    print(partial_stream.__str__(extended=True))
                    raise ex

                for gap in partial_stream.get_gaps():
                    self.logger.warning(
                        '%s: %d sample gap at %s'
                        % ('.'.join(gap[:4]), gap[7], gap[4]))

            if self.stream is None:
                self.stream = partial_stream
            elif partial_stream is not None:
                self.stream += partial_stream

            if cache and partial_stream is not None:
                input_file = '.'.join([self.make_cache_name(),
                                       self.cache_format])
                self.logger.info('Caching %s locally as: %s'
                                 % (self.cache_format, input_file))
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    try:
                        self.stream.write(input_file, format=self.cache_format)
                    except UserWarning as ex:
                        self.logger.warning(ex.args[0].replace('\n', ' '))

        if self.stream is not None:
            self.stream = self.stream.merge(fill_value=fill_value).sort()

        if not response:
            return

        if inventory_dataless is not None:
            self.logger.info('Reading responses from StationXML file: %s' %
                             inventory_dataless)
            inventory = dataless2inventory(inventory_dataless)
        else:
            inventory = None

        for client in self.clients:
            if 'station' not in client.services:
                continue

            if inventory is None:
                remaining = stations
            else:
                remaining = [station for station in stations
                             if len(inventory.select(station=station)) == 0]
            if len(remaining) == 0:
                break

            self.logger.info('Trying FDSN server: ' + client.base_url)
            self.logger.info(
                'Requesting inventory: ' + ', '.join(remaining))
            try:
                partial_inventory = client.get_stations(
                    network=','.join(networks),
                    station=','.join(remaining),
                    location=','.join(locations),
                    channel=','.join(channels),
                    startbefore=start, endafter=end,
                    level='response', includerestricted=True)

                if inventory is None:
                    inventory = partial_inventory
                else:
                    inventory += partial_inventory

            except FDSNException as ex:
                partial_inventory = None
                self.logger.warning(fdsn_error_message(ex))

            if cache and partial_inventory is not None:
                inventory_dataless = self.make_cache_name() + '.xml'
                self.logger.info('Caching StationXML locally as: %s'
                                 % inventory_dataless)
                inventory.write(inventory_dataless, format='STATIONXML')

        if self.stream is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                try:
                    not_found = self.stream.attach_response(inventory)
                    if len(not_found) > 0:
                        self.logger.warning(
                            'No response found:' +
                            ', '.join([trace.id for trace in not_found]))
                except (UserWarning, ValueError) as ex:
                    self.logger.warning(ex.args[0].replace('\n', ' '))

            remaining = [station for station in stations
                         if station not in set([trace.stats.station
                                                for trace in self.stream])]
            if len(remaining) > 0:
                self.logger.warning(
                    'Missing stations: ' + ', '.join(remaining))

            for trace in self.stream:
                if trace.id in inventory.get_contents()['channels']:
                    trace.stats.coordinates = \
                        inventory.get_coordinates(trace.id)
                else:
                    site = self.get_site(trace.stats.station)
                    if site is not None:
                        trace.stats.coordinates = AttribDict({
                            'latitude': site.lat,
                            'longitude': site.lon,
                            'elevation': site.elev,
                            'local_depth': 0.0,  # get_site needs work here
                            })

                missing_attributes = [
                    attribute for attribute in ['response', 'coordinates']
                    if attribute not in trace.stats]
                if missing_attributes:
                    self.logger.warning(
                        'No %s for: %s' %
                        (', '.join(missing_attributes), trace.id))

        else:
            self.logger.warning('No data loaded.')

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
                start_string = short_utc(np.max([trace.stats.starttime
                                                 for trace in self.stream]))

            common_name = factor_names(self.stream)[0]

            file_parts += [common_name, start_string]
        else:
            if hasattr(self, 'file'):
                calibration_name = self.info['file'].replace('.gz', '')
                calibration_name = calibration_name.replace('.wav', '')
                file_parts += [calibration_name]

        file_name = '_'.join(file_parts) + '.png'

        self.logger.info('Saving to', file_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    def plot_stream(self, save=False):
        '''
        Quick plot of signals.
        '''
        self.stream.plot(handle=True)

        if save:
            self.save_image()

    def _save_figure(self, obj, label, savefig):

        if savefig:
            if not isinstance(obj, plt.Figure):
                obj = obj.get_figure()
            label = label.replace(' ', '_').replace(':', '-')
            file_name = '_'.join([label, self.name]) + '.png'
            obj.savefig(file_name, dpi=300, bbox_inches='tight')
