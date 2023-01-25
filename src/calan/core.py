"""
A collection of utilities useful for station quality analysis.
"""
# pylint: disable=too-many-lines
from copy import deepcopy
import os
import re
import queue
import inspect
import logging
from logging import getLogger
from logging.config import dictConfig
from glob import glob
from time import time
from tempfile import gettempdir
from operator import attrgetter

import requests
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy.signal as sp
from scipy.signal import lti, ZerosPolesGain

from obspy import read_inventory, UTCDateTime
from obspy.clients import fdsn
from obspy.core.inventory import CoefficientsTypeResponseStage
from obspy.core.event import (
    Pick, Arrival, Amplitude, StationMagnitude, WaveformStreamID)

from calan.utilities import string_list
from calan import chis_archive

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
PACKAGE = os.path.basename(os.path.dirname(__file__))
VERSION = '1.1.0'

CHIS_FDSN_SERVERS = (
    'http://fdsn.seismo.nrcan.gc.ca',  # production, SeisComP3
    'http://sc3-stage.seismo.nrcan.gc.ca',  # staging, seisComP3
)
DEFAULT_FDSN_SERVERS = tuple(list(CHIS_FDSN_SERVERS) + ['IRIS'])

NSLC = ['network', 'station', 'location', 'channel']
NSLCSE = NSLC + ['start', 'end']
GAP_COLUMNS = ['starttime', 'endtime', 'duration', 'samples']

NETWORK_KEYS = ((
    ('code', 'network'),
    ('description', 'network_description'),
))
STATION_KEYS = ((
    ('code', 'station'),
    ('site.name', 'site_name'),
    ('creation_date.datetime', 'creation_date'),
))
CHANNEL_KEYS = ((
    ('location_code', 'location'),
    ('code', 'channel'),
    ('latitude', 'latitude'),
    ('longitude', 'longitude'),
    ('elevation', 'elevation'),
    ('depth', 'depth'),
    ('azimuth', 'azimuth'),
    ('dip', 'dip'),
    ('sensor.description', 'sensor'),
    ('data_logger.description', 'data_logger'),
    ('sample_rate', 'sample_rate'),
    ('restricted_status', 'restricted_status'),
    ('start_date.datetime', 'start_date'),
    ('end_date.datetime', 'end_date',),
))


def get_clients(servers=None, test_timeout=2):
    """Return a list of FDSN clients given a list of URLs."""
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
                        'CHIS archive %s not available. Ignoring.', server)
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
    """Clean up certain obspy.clients.fdsn exception messages."""
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
    print(f'WARNING: StationXML converter "{STATIONXML_CONVERTER_FILE}" not found. '
          'Cannot convert dataless2inventory ')


def dataless2inventory(inventory_dataless, inventory_source='GSC'):
    """
    Convert a dataless SEED to an ObsPy Inventory.

    A StationXML file is produced in the same directory as the input file.

    Note
    ----
    This should be deprecated; ObsPy does it natively.
    """
    inventory_xml = dataless2stationxml(inventory_dataless,
                                        inventory_source=inventory_source)
    return read_inventory(inventory_xml)


def dataless2stationxml(inventory_dataless, inventory_source='GSC'):
    """
    Convert a dataless SEED to StationXML.

    A StationXML file is produced in the same directory as the input file.

    Note
    ----
    This should be deprecated; ObsPy does it natively.
    """
    logger = get_logger(__name__)
    if not os.path.isfile(inventory_dataless):
        logger.warning('Dataless SEED file "%s" not found', inventory_dataless)

    inventory_xml = inventory_dataless.replace('.dataless', '.xml')

    if not os.path.isfile(inventory_xml):
        os.system(
            f'java -jar {STATIONXML_CONVERTER} --xml --prettyprint --source '
            f'{inventory_source} --output {inventory_xml} {inventory_dataless}')

    return inventory_xml


def inventory2dataless(inventory_xml):
    """Convert StationXML inventory to dataless SEED."""
    logger = get_logger(__name__)
    if not os.path.isfile(inventory_xml):
        logger.warning('StationXML file "%s" not found', inventory_xml)

    inventory_dataless = inventory_xml.replace('.xml', '.dataless')

    if not os.path.isfile(inventory_dataless):
        os.system(
            f'java -jar {STATIONXML_CONVERTER} --seed --output '
            f'{inventory_dataless} {inventory_xml}')

    return inventory_dataless


def _elapsed_since(tick):
    return str(pd.to_timedelta(round(time() - tick), 's')).split()[-1]


def get_chis_stations(level='response', minlatitude=35, maxlatitude=90,
                      maxlongitude=-40, minlongitude=-170):
    """
    Return an inventory of all stations for which CHIS has waveform data.

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
    """
    logger = get_logger(__name__)

    args, _, _, defaults = inspect.getfullargspec(get_chis_stations)[:4]
    inventory_file = os.path.join(
        gettempdir(),
        '_'.join(f'{arg}{default}'
                 for arg, default in zip(args, defaults)) + '.xml')
    inventory_file = inventory_file.replace('level', '')

    if os.path.isfile(inventory_file):
        logger.info('Cache: %s', inventory_file)
        tick = time()
        inventory = read_inventory(inventory_file)
        logger.info('Elapsed: %s', _elapsed_since(tick))
    else:
        try:
            chis_fdsn_client = fdsn.Client(DEFAULT_FDSN_SERVERS[0])
        except fdsn.client.FDSNException:
            chis_fdsn_client = fdsn.Client(DEFAULT_FDSN_SERVERS[2])

        logger.info('Client: %s', chis_fdsn_client.base_url)

        tick = time()
        inventory = chis_fdsn_client.get_stations(
            level=level, minlatitude=minlatitude, maxlatitude=maxlatitude,
            maxlongitude=maxlongitude, minlongitude=minlongitude)
        logger.info(
            'Read %s inventory via %s: %s',
            inventory.sender.upper(), inventory.source, _elapsed_since(tick))

        tick = time()
        inventory.write(inventory_file, format='StationXML')
        logger.info(
            'Cached %s: %s', inventory_file, _elapsed_since(tick))


def sort_complex(array: np.ndarray) -> np.ndarray:
    """Sort complex array by absolute value, then by imaginary part."""
    return np.array(sorted(sorted(array, key=np.imag), key=np.abs))


def sensitivity(system: lti, f: float = 1) -> float:
    """Compute sensitivity at given frequency."""
    return np.abs(system.freqresp(w=2*np.pi*f)[1][0])


def gain_db(values: ArrayLike) -> np.ndarray:
    """Return transfer function gain in dB, given complex values."""
    return 20*np.log10(np.abs(values))


def phase_deg(values: ArrayLike) -> np.ndarray:
    """Return transfer function phase in degrees, given complex values."""
    return np.angle(np.array(values), deg=True)


def zpk_cancel(
    old: ZerosPolesGain,
    tolerance: float = 0,
    f_norm: float = 1,
    method: str = 'damping',
):
    """
    Remove (nearly) identical zero-pole pairs from a transfer function.

    The 'octave' method is that used in octave :func:`~control.minreal` and
    python-control :func:`~TransferFunction.minreal`.
    This method only considers the absolute distance between each
    pole `p` and zero `z`:

        `abs(z-p) <= tolerance`

    The 'damping' method is an improvement whereby pole-zero pairs must be
    farther apart in order to be considered cancelling when the damping is low.
    In this case poles and zeros are considered to cancel when the following
    condition is met:

        `abs(z-p)/sqrt(|z|*|p|)/sqrt(cos(angle(z))*cos(angle(p))) <= tolerance`

    Parameters
    ----------
        - old: transfer function before cancellation
        - tolerance: tolerance for cancellation
        - f_norm: normalization frequency - anywhere mid-band
        - method: 'damping' or 'octave' as described above
    """
    assert tolerance >= 0
    assert f_norm >= 0
    assert method in ['damping', 'octave']

    if len(old.zeros) == 0 or len(old.poles) == 0:
        return deepcopy(old)

    zeros = old.zeros.reshape(-1, 1)
    poles = old.poles.reshape(1, -1)

    condition = np.abs(zeros - poles)
    if method == 'damping':
        with np.errstate(divide='ignore', invalid='ignore'):
            condition /= (np.sqrt(np.abs(poles)*np.abs(zeros)) /
                          np.sqrt(np.cos(np.angle(poles)) *
                                  np.cos(np.angle(zeros))))
        # correct NaNs produced by 0/0
        condition[zeros == poles] = 0

    # retain only closest cancellation
    cancels = np.full_like(condition, False)
    while (condition <= tolerance).any():
        indices = np.unravel_index(np.nanargmin(condition), condition.shape)
        condition[indices[0], :] = np.Inf
        condition[:, indices[1]] = np.Inf
        cancels[indices[0], indices[1]] = True
        condition = np.ma.array(condition, mask=cancels)

    z_new = sort_complex(old.zeros[~cancels.any(axis=1)])
    p_new = sort_complex(old.poles[~cancels.any(axis=0).T])

    new = ZerosPolesGain(z_new, p_new, old.to_tf().num[0])

    return new


def zpk_divide(
    num: ZerosPolesGain,
    den: ZerosPolesGain,
    tolerance: float = 0,
) -> ZerosPolesGain:
    """Divide numerator by denominator, including pole-zero cancellation."""
    zeros = np.hstack((num.zeros, den.poles))
    zeros.sort()
    poles = np.hstack((num.poles, den.zeros))
    poles.sort()
    gain = num.gain/den.gain

    return zpk_cancel(ZerosPolesGain(zeros, poles, gain), tolerance)


def zpk_multiply(
    first: ZerosPolesGain,
    second: ZerosPolesGain,
    tolerance: float = 0,
) -> ZerosPolesGain:
    """Multiply first by second, including pole-zero cancellation."""
    zeros = np.hstack((first.zeros, second.zeros))
    zeros.sort()
    poles = np.hstack((first.poles, second.poles))
    poles.sort()
    gain = first.gain*second.gain

    return zpk_cancel(ZerosPolesGain(zeros, poles, gain), tolerance)


def flip(ndarray, axis):
    """
    Reverse order of elements in an array along the given axis.

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
    """
    if not hasattr(ndarray, 'ndim'):
        ndarray = np.asarray(ndarray)
    indexer = [slice(None)] * ndarray.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError as ex:
        raise ValueError(
            f'axis={axis} is invalid for {ndarray.ndim}-dimensional input'
        ) from ex
    return ndarray[tuple(indexer)]


def unwrap_mid(phase_in, f_in, f_midband=1, axis=-1, discont=np.pi):
    """
    Unwrap phase data in the range starting at midband.

    Unwrapping is done from -discont to discont starting at a specified
    midband frequency and working outwards.

    Parameters
    ----------
    phase_in: :class:`~numpy.array` or list
        Phase data.
    f_in: :class:`~numpy.array` or list
        Frequencies corresponding to phases.
    f_midband: float, optional
        Midband frequency at which to start unwrapping. Default is 1.
    axis: int, optional
        Axis along which to unwrap. Default is last axis.
    discont: float, optional
        Maximum value at which to unwrap discontinueties. Default is pi.

    Returns
    -------
    phase_out: :class:`~numpy.array`
        Unwrapped phase data.
    """
    assert f_in.ndim == 1

    i_mid = np.argmin(np.abs(np.array(f_in)/f_midband - 1))
    phase_below = phase_in.take(np.arange(i_mid), axis)
    phase_above = phase_in.take(np.arange(i_mid, phase_in.shape[axis]), axis)

    phase_below = flip(
        np.unwrap(flip(phase_below, axis), discont=discont, axis=axis), axis)
    phase_above = np.unwrap(phase_above, discont=discont, axis=axis)

    return np.concatenate((phase_below, phase_above), axis)


def zpk_from_zpsf(zeros, poles, sens, frequency):
    """
    Generate a LinearTimeInvariant model.

    Inputs are zeros, poles, sensitivity and frequency at which sensitivity is
    specified.
    """
    model = ZerosPolesGain(zeros, poles, 1)
    midband = abs(model.freqresp(2*np.pi*frequency)[1])
    return ZerosPolesGain(
        sort_complex(zeros), sort_complex(poles), sens/midband)


def long_names(stream, parts=tuple(NSLC), widths=(2, 5, 2, 3)):
    """Construct a list of names for the traces in a stream."""
    return ['.'.join([('%' + str(width) + 's') % trace.stats[part]
                      for part, width in zip(parts, widths)])
            for trace in stream]


def factor_names(stream):
    """
    Return a tuple with the factored names for the traces in a stream.

    The first item in the tuple is the part which is common to all traces;
    the second item in the tuple is a list of the parts which differ.
    """
    if len(stream) == 1:
        return stream[0].id[:-1], stream[0].id[-1]
    full_names = long_names(stream)
    sames = [letters[1:] == letters[:-1] for letters in zip(*full_names)]
    short_names = [''.join(letter for letter, same in zip(name, sames)
                           if not same) for name in full_names]
    common_name = ''.join(letter for letter, same in zip(full_names[0], sames)
                          if same)
    common_name = '.'.join(part.strip() for part in common_name.split('.'))
    return common_name, short_names


def recompute_normalization_factors(response, rtol=0.0002):
    """Compare stage normalization factors to computed values."""
    logger = getLogger(__name__)
    for stage in response.response_stages:
        try:
            normalization_factor = stage.normalization_factor
        except AttributeError:
            continue
        stage_lti = zpk_from_zpsf(
            stage.zeros, stage.poles, stage.stage_gain,
            stage.normalization_frequency)
        stage_gain = sp.freqresp(
            stage_lti,
            2*np.pi*stage.normalization_frequency)[1][0]
        adjustment = np.abs(stage_gain)/abs(stage.stage_gain)
        if np.isnan(adjustment):
            logger.error('Failed to calculate normalization factor adjustment')
        else:
            stage.normalization_factor *= adjustment
        if np.isclose(stage.normalization_factor,
                      normalization_factor, rtol=rtol):
            continue
        logger.warning(
            'Stage %d normalization factor %.6g in file differs from '
            'recalculated value %.6g by more than %g%%.',
            stage.stage_sequence_number, normalization_factor,
            stage.normalization_factor, 100*rtol)


def compute_decim_delay(b_stages, factors):
    """
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

    Example
    -------
        n_pad_upsample = compute_decim_delay(b_stages, factors)
    """
    n_pad_upsample = 0
    for i in reversed(range(len(factors))):
        if len(b_stages[i]) % 2 == 0:
            raise TypeError(
                'Decimation filters must be odd-order: '
                f'stage {i} has {len(b_stages[i])} coefficients.')
        n_pad_upsample = n_pad_upsample*factors[i] + len(b_stages[i]) - 1

    return n_pad_upsample


# pylint: disable=too-many-arguments, too-many-locals
def multi_decim(sig_in, b_stages, factors, z_in=0, sig_leftover=(),
                discard_initial=True):
    """
    Apply cascaded FIR filter and decimation stages to a signal.

    Example
    -------
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
    """
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
    """Extract decimation factors and filter coefficients from a list of stages."""
    logger = get_logger(__name__)
    b_stages = []
    factors = []
    for stage in stages:
        if (isinstance(stage, CoefficientsTypeResponseStage) and
                stage.decimation_factor > 1):
            if not factors:
                logger.info('Input sample rate %g sps',
                            stage.decimation_input_sample_rate)

            factors.append(stage.decimation_factor)
            b_stages.append(stage.numerator)
            logger.info(
                'Filter with %d coefficients and decimate by %d to %g sps',
                len(stage.numerator), stage.decimation_factor,
                stage.decimation_input_sample_rate/stage.decimation_factor)
    n_pad_upsample = compute_decim_delay(b_stages, factors)
    logger.info('Filtering and decimation by %d consumes %d samples',
                np.prod(factors), n_pad_upsample)

    return b_stages, factors


def truncnorm_shape(mean, std, clip_b, clip_a=None):
    """
    Convert mean, standard deviation and clip levels to shape parameters.

    See :class:`~scipy.stats.truncnorm'.

    :returns: a, b
    """
    if clip_a is None:
        clip_a = -clip_b
    shape_a, shape_b = (clip_a - mean) / std, (clip_b - mean) / std

    return shape_a, shape_b


def subplots_squeeze(fig, hspace=None, wspace=None):
    """
    Squeeze space ticks and ticklabels from between axes.

    For now this just supports the case of multiple axes stacked vertically,
    removing space between them and removing tick labels which would overlap.
    """
    try:
        axes_indices = [[ax.get_subplotspec().colspan.start,
                         ax.get_subplotspec().rowspan.start]
                        for ax in fig.axes]
    except AttributeError:
        axes_indices = [[ax.get_subplotspec().get_rows_columns()[4],
                         ax.get_subplotspec().get_rows_columns()[2]]
                        for ax in fig.axes]

    num_cols, num_rows = np.max(axes_indices, axis=0) + 1
    axes = np.reshape(fig.axes, (num_rows, num_cols))

    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if hspace < 0.05 and num_rows > 1:
        for i, ax in enumerate(axes[:, 0]):
            if i > 0:
                ax.yaxis.get_major_ticks()[-1].label.set_visible(False)
            if i < num_rows - 1:
                ax.yaxis.get_major_ticks()[0].label.set_visible(False)


def _missing_samples(delta, sampling_rate):
    return np.rint(np.fabs(delta)*sampling_rate)


def is_complete(stream, trace_ids=(), start=pd.Timestamp(0),
                end=pd.Timestamp.now(), tolerance=0.5):
    """Lightweight test whether stream is complete."""
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


def gap_list(stream, trace_ids=(), start=pd.Timestamp(0),
             end=pd.Timestamp.now(), tolerance=0.5):
    """
    Construct a dataframe of gaps, including start & end gaps.

    If an empty stream is provided the returned value are empty dataframes
    with the correct columns.

    Returns
    -------
    2-tuple of pd.DataFrame:
        gaps_df, overlap_df
    """
    if trace_ids is None:
        trace_ids = ()
    if not isinstance(start, pd.Timestamp):
        start = pd.to_datetime(start.datetime)
    if not isinstance(end, pd.Timestamp):
        end = pd.to_datetime(end.datetime)

    if stream:
        gaps_df = pd.DataFrame(stream.get_gaps(),
                               columns=NSLC + GAP_COLUMNS)

        gaps_df.insert(
            0, 'id', ['.'.join(items)
                      for _, items in gaps_df[NSLC].iterrows()])
        gaps_df['sampling_rate'] = np.round(gaps_df.samples/gaps_df.duration)
        gaps_df.starttime = gaps_df.starttime.apply(
            lambda item: pd.to_datetime(item.datetime))
        gaps_df.endtime = gaps_df.endtime.apply(
            lambda item: pd.to_datetime(item.datetime))
    else:
        gaps_df = pd.DataFrame(
            columns=['id'] + NSLC + GAP_COLUMNS + ['sampling_rate'])

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
    """Compute fraction of requested data which is available."""
    trace_ids = string_list(trace_ids)

    expected_duration = len(trace_ids)*((UTCDateTime(end) -
                                         UTCDateTime(start)))
    gap_duration = gaps_df.loc[gaps_df.id.isin(trace_ids)].duration.sum()
    if not expected_duration:
        return np.NaN

    return 1 - gap_duration/expected_duration


def log_availability(logger, gaps_df, trace_ids, start, end,
                     column='duration'):
    """Summarize availability to a log file, given a gap listing."""
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
        '%s was %.1f%% complete with %d gap(s), e.g.:',
        common_id, percent_available, num_gaps)

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
            gap_start = str(gap.starttime.time())[:-3]
        else:
            gap_start = str(gap.starttime)[:-3]
        logger.info(
            f'{gap.id}: {gap_start} start of {gap[column]:3g} {gap_unit} gap '
            f'({gap.note})')


def inventory_items(inventory):
    """Iterate through network, station, channel of an inventory."""
    for network in inventory:
        for station in network:
            for channel in station:
                yield network, station, channel


def inventory_stations(inventory):
    """Iterate through network, station of an inventory."""
    for network in inventory:
        for station in network:
            yield network, station


def inventory2df(inventory):
    """Summarize obspy.Inventory in pandas.DataFrame."""
    def get(key, item):
        try:
            return attrgetter(key)(item)
        except AttributeError:
            return None

    df = pd.DataFrame()
    for key, column in NETWORK_KEYS:
        df[column] = [
            get(key, network) for network, _, _ in inventory_items(inventory)]
    for key, column in STATION_KEYS:
        df[column] = [
            get(key, station) for _, station, _ in inventory_items(inventory)]
    for key, column in CHANNEL_KEYS:
        df[column] = [
            get(key, channel) for _, _, channel in inventory_items(inventory)]

    for column in df.columns.values:
        if column.endswith('date'):
            df[column] = pd.to_datetime(df[column])

    df.dropna(axis='columns', how='all', inplace=True)
    df.set_index(NSLC, inplace=True)

    return df


def channels2df(inventory):
    """
    Create table of channels in inventory.

    Parameters
    ----------
    inventory : obspy.Inventory
        station inventory

    Returns
    -------
    df : pandas.DataFrame
        station table with index 'network', 'station','location' 'channel'
        and columns 'start' and 'end'.
    """
    # TODO: consider merging contiguous time ranges, but carefully!
    df = pd.DataFrame()
    df['network'] = [network.code
                     for network, _, _ in inventory_items(inventory)]
    df['station'] = [station.code
                     for _, station, _ in inventory_items(inventory)]
    df['location'] = [channel.location_code
                      for _, _, channel in inventory_items(inventory)]
    df['channel'] = [channel.code
                     for _, _, channel in inventory_items(inventory)]

    df['start'] = pd.to_datetime([
        channel.start_date.datetime if channel.start_date else pd.NaT
        for _, _, channel in inventory_items(inventory)])
    df['end'] = pd.to_datetime([
        channel.end_date.datetime if channel.end_date else pd.NaT
        for _, _, channel in inventory_items(inventory)])

    df['latitude'] = [station.latitude
                      for _, station, _ in inventory_items(inventory)]
    df['longitude'] = [station.longitude
                       for _, station, _ in inventory_items(inventory)]
    df['elevation_km'] = [channel.elevation * 1e-3
                          for _, _, channel in inventory_items(inventory)]
    df['depth_km'] = [channel.depth * 1e-3
                      for _, _, channel in inventory_items(inventory)]
    df['name'] = [station.site.name
                  for _, station, _ in inventory_items(inventory)]

    if df.duplicated(NSLCSE).any():
        getLogger(__name__).warning(
            'Keeping last of duplicate keys: %s',
            df.loc[df.duplicated(NSLCSE, keep=False)])
        df.drop_duplicates(NSLCSE, keep='last', inplace=True)
    df.set_index(NSLCSE, verify_integrity=True, inplace=True)
    df.sort_index(inplace=True)

    return df


def stations2df(inventory):
    """
    Create table of stations in inventory.

    Parameters
    ----------
    inventory : obspy.Inventory
        station inventory

    Returns
    -------
    df : pandas.DataFrame
        station table with index 'network', 'station','location' 'channel'
        and columns 'start' and 'end'.
    """
    # TODO: consider merging contiguous time ranges, but carefully!
    df = pd.DataFrame()
    df['network'] = [network.code
                     for network, _ in inventory_stations(inventory)]
    df['station'] = [station.code
                     for _, station in inventory_stations(inventory)]

    df['start'] = pd.to_datetime([
        station.start_date.datetime if station.start_date else pd.NaT
        for _, station in inventory_stations(inventory)])
    df['start'] = df['start'].dt.date
    df['end'] = pd.to_datetime([
        station.end_date.datetime if station.end_date else pd.NaT
        for _, station in inventory_stations(inventory)])
    df['end'] = df['end'].dt.date

    df['latitude'] = [station.latitude
                      for _, station in inventory_stations(inventory)]
    df['longitude'] = [station.longitude
                       for _, station in inventory_stations(inventory)]
    df['elevation_km'] = [station.elevation * 1e-3
                          for _, station in inventory_stations(inventory)]
    df['name'] = [station.site.name
                  for _, station in inventory_stations(inventory)]

    df.set_index(NSLCSE[:2] + NSLCSE[-2:], verify_integrity=True, inplace=True)
    df.sort_index(inplace=True)

    return df


def read_sql(file_name, parse_dates=('start', 'end'), index=(), dtype=None):
    """
    Read pipe-delimited SQL query result, ignoring non-pipe-delimited header.
    """
    if dtype is None:
        dtype = {'count': int}

    # determine structure of file
    skiprows = []
    found_header = False
    with open(file_name, encoding='UTF-8') as file:
        for i, line in enumerate(file):
            if '|' in line:
                pipes = np.array([match.start() for match in re.finditer(r'\|', line)])
                colspecs = list(zip([0] + list(pipes + 1), list(pipes) + [len(line)]))
                found_header = True
            else:
                skiprows.append(i)
            if set(line) == set('-+'):
                break
    if not found_header:
        raise RuntimeError('No header line found.')

    df = pd.read_fwf(file_name, sep='|', skiprows=skiprows,
                     colspecs=colspecs, parse_dates=list(parse_dates),
                     dtype=dtype)

    for column, dtype in dtype.items():
        if column in df and dtype == str:
            df[column] = df[column].fillna('')

    if index:
        df.set_index(list(index), inplace=True, verify_integrity=True)

    return df


# %% logging
FILE_NAME = os.path.basename(__file__)
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
LOG_SETTINGS = {
    'version': 1,  # logging schema
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'when': 'midnight',
            'utc': True,
            'filename': '',
            'level': 'DEBUG',
            'formatter': 'detailed',
        },
    },
    'formatters': {
        'simple': {
            'format':
            '%(levelname)-8s %(filename)s:%(name)s:%(funcName)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - '
                      '%(levelname)-8s %(filename)s:%(name)s:%(funcName)s - '
                      '%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
}

SIMPLE_LOG_SETTINGS = {
    'version': 1,  # logging schema
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': '',
            'level': 'DEBUG',
            'formatter': 'simple',
            'mode': 'w',
        },
    },
    'formatters': {
        'simple': {
            'format': '%(levelname)-8s %(message)s'
        },
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
}


def get_logger(name, log_file_name='', log_console_level='INFO'):
    """
    Return named logger which logs to console and file.

    Configures root logger for log_file_name and log_console_level.
    Logging to file is always at DEBUG level.
    """
    assert log_console_level in LOG_LEVELS

    handlers = logging.getLogger().handlers
    not_previously_configured = len(handlers) == 0

    if not_previously_configured:
        LOG_SETTINGS['handlers']['console'].update(
            {'level': log_console_level})
        LOG_SETTINGS['handlers']['file'].update(
            {'filename': log_file_name})
        dictConfig(LOG_SETTINGS)

    logger = logging.getLogger(name)
    if not_previously_configured:
        logger.info('Logfile: %s', os.path.abspath(log_file_name))
        logger.info('Package: %s v%s', PACKAGE, VERSION)
    return logger


class LoggerWriter:
    """
    Class for making a logger behave like a file.

    A typical usage is to us a LoggerWriter as an argument to
    contextlib.redirect_stdout() so that anything emitted to stdout
    (within python) is logged at the specified level.
    """

    def __init__(self, logger, level, name=None):
        self.logger = logger
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.level = level
        self.name = name

    def write(self, message):
        """Simulate file.write()."""
        if message != '\n':
            if self.name:
                message = self.name + ' - ' + message
            self.logger.log(self.level, message)

    def flush(self):
        """Simulate file.flush()."""


def get_channel(obj, event=None):
    """
    Return SEED string associated with ObsPy object.
    """
    waveform_id = get_waveform_id(obj, event)
    if not isinstance(waveform_id, WaveformStreamID):
        return ''

    return waveform_id.get_seed_string()


def get_waveform_id(obj, event=None):
    """
    Return waveform_id associated with ObsPy object.
    """
    if obj is None:
        return None
    if 'waveform_id' in obj and obj.waveform_id is not None:
        return obj.waveform_id

    pick = get_pick(obj, event)

    if not pick:
        return None

    return pick.waveform_id


def get_pick(obj, event=None):
    """
    Look up Pick associated with object.

    Arguments
    ---------
    obj: Pick, Arrival, Amplitude or StationMagnitude
        object in question
    event: Event
        not required for Arrival or Amplitude if referential integrity is
        intact.
    """
    if obj is None:
        return None

    if isinstance(obj, Pick):
        return obj

    if isinstance(obj, (Arrival, Amplitude)):
        if obj.pick_id is None:
            return None

        pick = obj.pick_id.get_referred_object()
        if pick is None:
            if not event:
                raise ValueError(
                    'Need Event to find Pick associated with this ' +
                    obj.__class__.__name__ + '.')
            pick = next((pick for pick in event.picks
                         if pick.resource_id == obj.pick_id), None)
        return pick

    if isinstance(obj, StationMagnitude):
        return get_pick(get_amplitude(obj, event), event)

    raise TypeError('object must be one of: Pick, Arrival, Amplitude, '
                    'StationMagnitude')


# pylint: disable=too-many-return-statements
def get_amplitude(obj, event=None):
    """
    Look up Amplitude associated with object.

    Arguments
    ---------
    obj: Pick, Arrival, Amplitude or StationMagnitude
        object in question
    event: Event
        not required for StationMagnitude if referential integrity is intact.
    """
    if obj is None:
        return None

    assert isinstance(obj, (Pick, Arrival, Amplitude, StationMagnitude))

    if isinstance(obj, Amplitude):
        return obj

    if isinstance(obj, Pick):
        return next((item for item in event.amplitudes
                     if item.pick_id == obj.resource_id), None)

    if isinstance(obj, StationMagnitude):
        if obj.amplitude_id is None:
            return None

        amplitude = obj.amplitude_id.get_referred_object()
        if amplitude is None:
            if not event:
                raise ValueError(
                    'Need Event find Amplitude associated with this ' +
                    obj.__class__.__name__ + '.')
            amplitude = next(
                (amplitude for amplitude in event.amplitudes
                 if amplitude.resource_id == obj.amplitude_id), None)
        return amplitude

    if isinstance(obj, Arrival):
        return get_amplitude(get_pick(obj, event), event)

    return None
