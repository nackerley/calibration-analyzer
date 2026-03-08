"""A collection of utilities useful for station quality analysis."""
# pylint: disable=too-many-lines
import os
import re
import sys
import logging
from copy import deepcopy
from datetime import datetime
from logging import getLogger, Logger
from logging.config import dictConfig
from typing import Iterator, Type
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray
import pandas as pd
import scipy.signal as sp
from scipy.signal import lti, ZerosPolesGain, TransferFunction, StateSpace
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from obspy import read_inventory, UTCDateTime, Stream
from obspy.core.inventory import (
    Inventory, Network, Station, Channel,
    ResponseStage, InstrumentSensitivity, PolesZerosResponseStage,
    CoefficientsTypeResponseStage, FIRResponseStage)
from obspy.core.event import \
    Pick, Arrival, Amplitude, StationMagnitude, Event, WaveformStreamID

from calan import PACKAGE_VERSION
from calan.utilities import string_list

CHIS_PUBLIC_FDSNWS = 'https://www.earthquakescanada.nrcan.gc.ca/'


NSLC = ['network', 'station', 'location', 'channel']
NSLCSE = NSLC + ['start', 'end']
GAP_COLUMNS = ['starttime', 'endtime', 'duration', 'samples']


def fdsn_error_message(ex: Exception) -> str:
    """Clean up certain obspy.clients.fdsn exception messages."""
    lines = ex.args[0].split('\n')
    if 'No data available' in lines[0]:
        msg = lines[0]
    else:
        msg = ' '.join(lines)
    return msg


STATIONXML_CONVERTER_FILE = 'stationxml-converter-1.0.9.jar'
STATIONXML_CONVERTER = Path(__file__).parent.joinpath(
    'jar', STATIONXML_CONVERTER_FILE)
if not STATIONXML_CONVERTER.exists():
    print(f'WARNING: StationXML converter "{STATIONXML_CONVERTER_FILE}" '
          'not found. Cannot convert dataless2inventory ')


def dataless2inventory(
    inventory_dataless: str,
    inventory_source: str = 'GSC',
) -> Inventory:
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


def dataless2stationxml(
    inventory_dataless: str,
    inventory_source: str = 'GSC',
) -> str:
    """
    Convert a dataless SEED to StationXML.

    A StationXML file is produced in the same directory as the input file.

    Note
    ----
    This should be deprecated; ObsPy does it natively.
    """
    logger = getLogger(__name__)
    if not Path(inventory_dataless).is_file():
        logger.warning('Dataless SEED file "%s" not found', inventory_dataless)

    inventory_xml = inventory_dataless.replace('.dataless', '.xml')

    if not Path(inventory_xml).is_file():
        os.system(
            f'java -jar {STATIONXML_CONVERTER} --xml --prettyprint --source '
            f'{inventory_source} --output {inventory_xml} '
            f'{inventory_dataless}')

    return inventory_xml


def inventory2dataless(inventory_xml: str) -> str:
    """Convert StationXML inventory to dataless SEED."""
    logger = getLogger(__name__)
    if not Path(inventory_xml).is_file():
        logger.warning('StationXML file "%s" not found', inventory_xml)

    inventory_dataless = inventory_xml.replace('.xml', '.dataless')

    if not Path(inventory_dataless).is_file():
        os.system(
            f'java -jar {STATIONXML_CONVERTER} --seed --output '
            f'{inventory_dataless} {inventory_xml}')

    return inventory_dataless


def sort_complex(array: NDArray) -> NDArray:
    """Sort complex array by absolute value, then by imaginary part."""
    return np.array(sorted(sorted(np.array(array), key=np.imag), key=np.abs))


def sensitivity(system: lti, f: float = 1) -> float:
    """Compute sensitivity at given frequency."""
    return np.abs(system.freqresp(w=2*np.pi*f)[1][0])


def gain_db(values: NDArray) -> NDArray:
    """Return transfer function gain in dB, given complex values."""
    return 20*np.log10(np.abs(values))


def phase_deg(values: NDArray) -> NDArray:
    """Return transfer function phase in degrees, given complex values."""
    return np.angle(np.array(values), deg=True)


def lti_minreal(
    old: lti,
    tolerance: float = 0,
    f_norm: float = 1,
    method: str = 'damping',
) -> lti:
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

    old_type = type(old)
    old = lti_convert(old, ZerosPolesGain)

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
        condition[indices[0], :] = np.inf
        condition[:, indices[1]] = np.inf
        cancels[indices[0], indices[1]] = True
        condition = np.ma.array(condition, mask=cancels)

    z_new = sort_complex(old.zeros[~cancels.any(axis=1)])
    p_new = sort_complex(old.poles[~cancels.any(axis=0).T])

    new = ZerosPolesGain(z_new, p_new, old.to_tf().num[0])

    return lti_convert(new, old_type)


def lti_divide(num: lti, den: lti, tolerance: float = 0) -> lti:
    """Divide numerator by denominator, including pole-zero cancellation."""
    num_type = type(num)
    num = lti_convert(num, ZerosPolesGain)
    den = lti_convert(den, ZerosPolesGain)

    zeros = np.hstack((num.zeros, den.poles))
    poles = np.hstack((num.poles, den.zeros))
    gain = num.gain/den.gain
    quotient = lti_minreal(ZerosPolesGain(zeros, poles, gain), tolerance)

    return lti_convert(quotient, num_type)


def lti_multiply(first: lti, second: lti, tolerance: float = 0) -> lti:
    """Multiply first by second, including pole-zero cancellation."""
    first_type = type(first)
    first = lti_convert(first, ZerosPolesGain)
    second = lti_convert(second, ZerosPolesGain)

    zeros = np.hstack((first.zeros, second.zeros))
    poles = np.hstack((first.poles, second.poles))
    gain = first.gain*second.gain
    product = lti_minreal(ZerosPolesGain(zeros, poles, gain), tolerance)

    return lti_convert(product, first_type)


def stage2zpk(stage: PolesZerosResponseStage) -> ZerosPolesGain:
    """
    Generate a LinearTimeInvariant model.

    Inputs are zeros, poles, sensitivity and frequency at which sensitivity is
    specified.
    """
    model = ZerosPolesGain(stage.zeros, stage.poles, 1)
    midband = abs(model.freqresp(2*np.pi*stage.normalization_frequency)[1])
    return ZerosPolesGain(
        sort_complex(stage.zeros), sort_complex(stage.poles),
        stage.stage_gain/midband)


def zpk_cascade(
    stages: list[ResponseStage],
    initial: ZerosPolesGain = ZerosPolesGain([], [], 1),
) -> ZerosPolesGain:
    """Cascade obspy response stages into a scipy ZerosPolesGain system."""
    zpk = initial
    for stage in stages:
        if isinstance(stage, PolesZerosResponseStage):
            zpk = lti_multiply(zpk, stage2zpk(stage))
        elif isinstance(stage, ResponseStage):
            zpk = lti_multiply(zpk, ZerosPolesGain([], [], stage.stage_gain))
        else:
            raise RuntimeError(
                f'Unsupported stage type: {type(stage)}')
    return zpk


def stage_units(
    stages: ResponseStage | list[ResponseStage] |
    InstrumentSensitivity | list[InstrumentSensitivity],
) -> dict[str, str]:
    """Clean up and return a dictionary of relevant units."""
    if isinstance(stages, (ResponseStage, InstrumentSensitivity)):
        stages = [stages]

    input_ = (stages[0].input_units.replace('COUNTS', 'counts')
              .replace('M', 'm').replace('S', 's'))
    output = (stages[-1].output_units.replace('COUNTS', 'counts')
              .replace('M', 'm').replace('S', 's'))

    if '/' in input_:
        forward = f'{output}/({input_})'
    else:
        forward = f'{output}/{input_}'
    if '/' in output:
        reverse = f'{input_}/({output})'
    else:
        reverse = f'{input_}/{output}'

    return {
        'input': input_,
        'output': output,
        'forward': forward,
        'reverse': reverse,
    }


def lti_convert(system: lti, to_type: Type) -> lti:
    """Convert system to specified type."""
    if isinstance(system, to_type):
        return system
    if to_type == type(ZerosPolesGain):
        return system.to_zpk()
    if to_type == type(TransferFunction):
        return system.to_tf()
    if to_type == type(StateSpace):
        return system.to_ss()


def lti_is_proper(system: lti) -> bool:
    """Indicate whether transfer function is proper."""
    if not isinstance(system, TransferFunction):
        system = system.to_tf()

    return len(system.den) >= len(system.num)


def unwrap_mid(
    phase_in: NDArray[np.float64],
    f_in: NDArray[np.float64],
    f_midband: float = 1,
    axis: int = -1,
    discont: float = np.pi,
) -> NDArray[np.float64]:
    """
    Unwrap phase data in the range starting at midband.

    Unwrapping is done from -discont to discont starting at a specified
    midband frequency and working outwards.

    Arguments:
      - `phase_in`: Wrapped phase data.
      - `f_in`: Frequencies corresponding to phases.
      - `f_midband`: Midband frequency at which to start unwrapping.
      - `axis`: Axis along which to unwrap.
      - `discont`: Maximum value at which to unwrap discontinueties.
    """
    f_in = np.array(f_in)
    phase_in = np.array(phase_in)
    if f_in.ndim != 1:
        raise ValueError('Only 1D arrays of frequencies supported')

    i_mid = np.argmin(np.abs(np.array(f_in)/f_midband - 1))
    phase_below = phase_in.take(np.arange(i_mid), axis)
    phase_above = phase_in.take(np.arange(i_mid, phase_in.shape[axis]), axis)

    phase_below = np.flip(
        np.unwrap(np.flip(phase_below, axis), discont=discont, axis=axis),
        axis)
    phase_above = np.unwrap(phase_above, discont=discont, axis=axis)

    return np.concatenate((phase_below, phase_above), axis)


def factor_names(stream: Stream) -> tuple[str, list[str]]:
    """
    Return a tuple with the factored names for the traces in a stream.

    The first item in the tuple is the part which is common to all traces;
    the second item in the tuple is a list of the parts which differ.
    """
    if len(stream) == 1:
        return stream[0].id[:-1], stream[0].id[-1]
    ids = [trace.id for trace in stream]
    sames = [letters[1:] == letters[:-1] for letters in zip(*ids)]
    short_names = [''.join(letter for letter, same in zip(name, sames)
                           if not same) for name in ids]
    common_name = ''.join(letter for letter, same in zip(ids[0], sames)
                          if same)
    common_name = '.'.join(part.strip() for part in common_name.split('.'))
    return common_name, short_names


def compute_decim_delay(
    b_stages: list[NDArray[np.float64]],
    factors: list[int],
) -> int:
    """
    Determine total filter delay for multi-stage decimation.

    Each decimation filter must be odd-order so that the total filter delay
    is an integer number of samples. The result is the total number of extra
    samples required to load the multi decimation filters.  Actual
    filter delay, in seconds, for a symmetric filter, is:
        tDelay = n_pad_upsample/2/f_upsample
    where the initial sample rate is f_upsample in Hz.

    Parameters:
        - b_stages: decimation filter coefficients
        - factors: decimation factors for each stage

    Returns number of samples needed to load filters.
    """
    n_pad_upsample = 0
    for i, b_stage in reversed(list(enumerate(b_stages))):
        if len(b_stage) % 2 == 0:
            raise TypeError(
                'Decimation filters must be odd-order: '
                f'stage {i} has {len(b_stage)} coefficients.')
        n_pad_upsample = n_pad_upsample*factors[i] + len(b_stage) - 1

    return n_pad_upsample


# pylint: disable=too-many-arguments, too-many-locals
def multi_decim(
    sig_in: NDArray[np.float64],
    b_stages: list[NDArray[np.float64]],
    factors: list[int],
    z_in: float | list[NDArray[np.float64]] = 0,
    sig_leftover: NDArray[np.float64] = np.array(()),
    discard_initial: bool = True,
) -> tuple[NDArray[np.float64],
           list[NDArray[np.float64]],
           NDArray[np.float64]]:
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

    Returns output samples, filter delays and unused input samples.
    """
    if len(b_stages) != len(factors):
        raise ValueError(
            f'Number of coefficients lists {len(b_stages)} must equal '
            f'the number of decimation factors {len(factors)}.')

    if isinstance(z_in, (float, int)):
        z_in = [z_in*sp.lfilter_zi(b_stage, 1) for b_stage in b_stages]

    if len(z_in) != len(factors):
        raise ValueError(
            f'Number of initial stage delays {len(z_in)} must equal the '
            f'number of decimation factors {len(factors)}.')

    num_coeffs = [len(b_stage) for b_stage in b_stages]
    num_delays = [len(z_stage) for z_stage in z_in]
    if not all(np.array(num_delays) == np.array(num_coeffs) - 1):
        raise ValueError(
            f'Lengths of initial stage delays {num_delays} must be one less '
            f'than those of stage coefficients {num_coeffs}')

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
        return np.array([]), [np.array(z_stage) for z_stage in z_in], sig_in

    # initialize signals and pass unused signal through to output
    sig_stages: list[NDArray[np.float64]] = [np.array([])]*(len(factors) + 1)
    sig_stages[0] = sig_in[:len_input_required]
    sig_unused = sig_in[len_input_required:]

    if discard_initial:
        first_indices = list(np.array(num_coeffs) - 1)
    else:
        first_indices = [0]*len(b_stages)

    # alternately filter and decimate according to stage specifications
    z_out = [np.array([])]*len(factors)
    for i, _ in enumerate(factors):
        sig_stages[i], z_out[i] = sp.lfilter(
            b_stages[i], 1, sig_stages[i], zi=z_in[i])
        sig_stages[i + 1] = sig_stages[i][first_indices[i]::factors[i]]

    if len(sig_stages[-1]) < 1:
        raise RuntimeError('Unexpected no samples after filtering.')

    if len(sig_stages[-1]) != len_output:
        raise RuntimeError(
            f'Expected {len_output} samples, got {len(sig_stages[-1])}.')

    return sig_stages[-1], z_out, sig_unused


def extract_decimation_coefficients(
    stages: list[ResponseStage]
) -> tuple[list[NDArray[np.float64]], list[int]]:
    """Extract decimation factors, filter coefficients from list of stages."""
    logger = getLogger(__name__)
    b_stages: list[NDArray[np.float64]] = []
    factors: list[int] = []
    for stage in stages:
        if stage.decimation_factor is not None and stage.decimation_factor > 1:
            if not factors:
                logger.info(
                    'Input sample rate %g sps',
                    stage.decimation_input_sample_rate)

            factors.append(stage.decimation_factor)
            if isinstance(stage, CoefficientsTypeResponseStage):
                if len(stage.denominator) != 0 or \
                        stage.cf_transfer_function_type != 'DIGITAL':
                    raise RuntimeError(
                        f'Analog decimation is nonsensical:\n{str(stage)}.')
                coeffs = stage.numerator
            elif isinstance(stage, FIRResponseStage):
                coeffs = stage.coefficients
                if stage.symmetry == 'EVEN':
                    coeffs = np.hstack((coeffs, np.flip(coeffs)))
                elif stage.symmetry == 'ODD':
                    coeffs = np.hstack((coeffs, np.flip(coeffs[:-1])))
            else:
                raise RuntimeError(
                    f'Decimation stage type not supported:\n{str(stage)}')
            b_stages.append(coeffs)

            logger.info(
                'Filter with %d coefficients and decimate by %d to %g sps',
                len(b_stages[-1]), stage.decimation_factor,
                stage.decimation_input_sample_rate/stage.decimation_factor)
    n_pad_upsample = compute_decim_delay(b_stages, factors)
    logger.info('Filtering and decimation by %d consumes %d samples',
                np.prod(factors), n_pad_upsample)

    return b_stages, factors


def truncnorm_shape(
    mean: float, std: float, clip_b: float, clip_a: float | None = None,
) -> tuple[float, float]:
    """
    Convert mean, standard deviation and clip levels to shape parameters.

    See :class:`~scipy.stats.truncnorm'.

    Returns shape parameters a, b.
    """
    if clip_a is None:
        clip_a = -clip_b
    shape_a, shape_b = (clip_a - mean) / std, (clip_b - mean) / std

    return shape_a, shape_b


def subplots_squeeze(
    fig: Figure,
    hspace: float | None = None,
    wspace: float | None = None,
) -> None:
    """
    Squeeze space ticks and ticklabels from between axes.

    For now this just supports the case of multiple axes stacked vertically,
    removing space between them and removing tick labels which would overlap.
    """
    def _get_row_col_start(
        subplotspec: SubplotSpec | None
    ) -> tuple[int, int]:
        if subplotspec is None:
            return (0, 0)

        return (subplotspec.colspan.start, subplotspec.rowspan.start)

    axes_indices = [
        _get_row_col_start(ax.get_subplotspec()) for ax in fig.axes]

    num_cols, num_rows = np.max(axes_indices, axis=0) + 1
    if num_cols == 0 or num_rows == 0:
        raise RuntimeError('This only works with subplots.')

    axes = np.reshape(np.array(fig.axes), (num_rows, num_cols))

    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if hspace and hspace < 0.05 and num_rows > 1:
        for i, ax in enumerate(axes[:, 0]):
            if i > 0:
                ax.yaxis.get_major_ticks()[-1].label.set_visible(False)
            if i < num_rows - 1:
                ax.yaxis.get_major_ticks()[0].label.set_visible(False)


def _missing_samples(delta: float, sampling_rate: float) -> int:
    return np.rint(np.fabs(delta)*sampling_rate)


def is_complete(
    stream: Stream,
    trace_ids: list[str] | None = None,
    start: datetime | str = pd.Timestamp(0),
    end: datetime | str = pd.Timestamp.now(),
    tolerance: float = 0.5
) -> bool:
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


def gap_list(
    stream: Stream,
    trace_ids: list[str] | None = None,
    start: datetime | UTCDateTime = pd.Timestamp(0),
    end: datetime | UTCDateTime = pd.Timestamp.now(),
    tolerance: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct a dataframe of gaps, including start & end gaps.

    If an empty stream is provided the returned value are empty dataframes
    with the correct columns.

    Returns aps_df, overlap_df.
    """
    if trace_ids is None:
        trace_ids = []
    if not isinstance(start, datetime):
        start = pd.to_datetime(start.datetime)
    if not isinstance(end, datetime):
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

    missing_data = []
    for trace_id in trace_ids:
        id_stream = stream.select(id=trace_id)
        if not id_stream:
            duration = (end - start).total_seconds()
            missing_data.append((
                trace_id,
                trace_id.split('.')[0],
                trace_id.split('.')[1],
                trace_id.split('.')[2],
                trace_id.split('.')[3],
                start,
                end,
                duration,
                -1,
                np.nan
            ))

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
            missing_data.append((
                trace_id,
                trace.stats.network,
                trace.stats.station,
                trace.stats.location,
                trace.stats.channel,
                start,
                trace_start,
                duration,
                _missing_samples(duration, sampling_rate),
                sampling_rate,
            ))

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
            missing_data.append((
                trace_id,
                trace.stats.network,
                trace.stats.station,
                trace.stats.location,
                trace.stats.channel,
                trace_end,
                end,
                duration,
                _missing_samples(duration, sampling_rate),
                sampling_rate,
            ))

    if missing_data:
        gaps_df = pd.concat((
            gaps_df,
            pd.DataFrame(missing_data, columns=gaps_df.columns)))
    gaps_df.sort_values(by=['starttime', 'endtime'],
                        ascending=[True, False], inplace=True)
    gaps_df.reset_index(inplace=True, drop=True)

    return gaps_df[gaps_df.duration > 0], gaps_df[gaps_df.duration <= 0]


def fraction_available(
    trace_ids: list[str],
    start: datetime | str,
    end: datetime | str,
    gaps_df: pd.DataFrame,
) -> float:
    """Compute fraction of requested data which is available."""
    trace_ids = string_list(trace_ids)

    expected_duration = len(trace_ids)*((UTCDateTime(end) -
                                         UTCDateTime(start)))
    gap_duration = gaps_df.loc[gaps_df.id.isin(trace_ids)].duration.sum()
    if not expected_duration:
        return np.nan

    return 1 - gap_duration/expected_duration


def log_availability(
    logger: Logger,
    gaps_df: pd.DataFrame,
    trace_ids: list[str],
    start: datetime,
    end: datetime,
    column: str = 'duration',
) -> None:
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


def inventory_items(
    inventory: Inventory,
) -> Iterator[tuple[Network, Station, Channel]]:
    """Iterate through network, station, channel of an inventory."""
    for network in inventory:
        for station in network:
            for channel in station:
                yield network, station, channel


def inventory_stations(
    inventory: Inventory,
) -> Iterator[tuple[Network, Station]]:
    """Iterate through network, station of an inventory."""
    for network in inventory:
        for station in network:
            yield network, station


def channel_table(inv: Inventory) -> pd.DataFrame:
    """Convert inventory to dataframe."""

    data = {
        'Network': [net.code for net, _, _ in inventory_items(inv)],
        'Station': [sta.code for _, sta, _ in inventory_items(inv)],
        'Location': [chn.location_code for _, _, chn in inventory_items(inv)],
        'Channel': [chn.code for _, _, chn in inventory_items(inv)],
        'Latitude': [chn.latitude for _, _, chn in inventory_items(inv)],
        'Longitude': [chn.longitude for _, _, chn in inventory_items(inv)],
        'Elevation': [chn.elevation for _, _, chn in inventory_items(inv)],
        'Depth': [chn.depth for _, _, chn in inventory_items(inv)],
        'Azimuth': [chn.azimuth for _, _, chn in inventory_items(inv)],
        'Dip': [chn.dip for _, _, chn in inventory_items(inv)],
        'SensorDescription': [
            chn.sensor.type for _, _, chn in inventory_items(inv)],
        'Scale': [
            chn.response.instrument_sensitivity.value
            for _, _, chn in inventory_items(inv)],
        'ScaleFreq': [
            chn.response.instrument_sensitivity.frequency
            for _, _, chn in inventory_items(inv)],
        'ScaleUnits': [
            chn.response.instrument_sensitivity.input_units
            for _, _, chn in inventory_items(inv)],
        'SampleRate': [chn.sample_rate for _, _, chn in inventory_items(inv)],
        'StartTime': [chn.start_date for _, _, chn in inventory_items(inv)],
        'EndTime': [chn.end_date for _, _, chn in inventory_items(inv)],
    }

    return pd.DataFrame(data).set_index(
        ['Network', 'Station', 'Location', 'Channel'])


def station_table(inv: Inventory) -> pd.DataFrame:
    """Convert inventory to dataframe."""

    data = {
        'Network': [net.code for net, _ in inventory_stations(inv)],
        'Station': [sta.code for _, sta in inventory_stations(inv)],
        'Latitude': [sta.latitude for _, sta in inventory_stations(inv)],
        'Longitude': [sta.longitude for _, sta in inventory_stations(inv)],
        'Elevation': [sta.elevation for _, sta in inventory_stations(inv)],
        'SiteName': [sta.site.name for _, sta in inventory_stations(inv)],
        'StartTime': [sta.start_date for _, sta in inventory_stations(inv)],
        'EndTime': [sta.end_date for _, sta in inventory_stations(inv)],
    }

    return pd.DataFrame(data).set_index(['Network', 'Station'])


def read_sql(
    file_name: str,
    parse_dates: tuple[str, ...] = ('start', 'end'),
    index: list[str] | None = None,
    dtypes: dict[str, DTypeLike] | None = None,
) -> pd.DataFrame:
    """
    Read pipe-delimited SQL query result.

    A non-pipe-delimited header is ignored.
    """
    if dtypes is None:
        dtypes = {'count': int}

    # determine structure of file
    skiprows = []
    found_header = False
    with open(Path(file_name).expanduser(), encoding='UTF-8') as file:
        for i, line in enumerate(file):
            if '|' in line:
                pipes = np.array([match.start()
                                  for match in re.finditer(r'\|', line)])
                colspecs = list(zip([0] + list(pipes + 1),
                                    list(pipes) + [len(line)]))
                found_header = True
            else:
                skiprows.append(i)
            if set(line) == set('-+'):
                break
    if not found_header:
        raise RuntimeError('No header line found.')

    df = pd.read_fwf(
        file_name, sep=r'\s+\|\s+', skiprows=skiprows, dtype=dtypes,
        colspecs=colspecs)  # pylint: disable=possibly-used-before-assignment
    for col in parse_dates:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    for column, dtype in dtypes.items():
        if column in df and dtype == str:
            df[column] = df[column].fillna('')

    if index is not None:
        df.set_index(list(index), inplace=True, verify_integrity=True)

    return df


# logging
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
LOG_SETTINGS: dict[str, int | dict[
    str, dict[str, str | bool | list[str]]]] = {
    'version': 1,  # schema
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
            'formatter': 'detailed',
        },
    },
    'formatters': {
        'simple': {
            'format':
            '%(levelname)-8s %(funcName)s - %(message)s'
        },
        'detailed': {
            'format': '%(levelname)-8s %(filename)s:%(name)s:%(funcName)s - '
                      '%(message)s',
        },
    },
    'loggers': {
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
}


def start_logger(
    name: str,
    log_file_name: str,
    log_console_level: str,
) -> Logger:
    """
    Return named logger which logs to console and given file.

    Configures root logger for log_file_name and log_console_level.
    Logging to file is always at DEBUG level.
    """
    assert log_console_level in LOG_LEVELS

    try:
        Path(log_file_name).unlink()
    except OSError:
        pass

    LOG_SETTINGS['handlers']['console'].update(  # type: ignore
        {'level': log_console_level})
    LOG_SETTINGS['handlers']['file'].update(  # type: ignore
        {'filename': log_file_name})
    dictConfig(LOG_SETTINGS)

    logger = logging.getLogger(name)

    logger.info('%s: %s', log_file_name, PACKAGE_VERSION)

    return logger


class LoggerWriter:
    """
    Class for making a logger behave like a file.

    A typical usage is to us a LoggerWriter as an argument to
    contextlib.redirect_stdout() so that anything emitted to stdout
    (within python) is logged at the specified level.
    """

    def __init__(
        self,
        logger: Logger,
        level: int | str,
        name: str | None = None,
    ) -> None:
        """Construct object."""
        self.logger = logger
        if isinstance(level, str):
            level = int(getattr(logging, level.upper()))
        self.level = level
        self.name = name

    def write(self, message: str) -> None:
        """Simulate file object write method."""
        if message != '\n':
            if self.name:
                message = self.name + ' - ' + message
            self.logger.log(self.level, message)

    def flush(self) -> None:
        """Simulate file object write method."""


def get_channel(
    obj: Pick | Arrival | Amplitude | StationMagnitude,
    event: Event | None = None,
) -> str:
    """Return SEED string associated with ObsPy object."""
    waveform_id = get_waveform_id(obj, event)
    if not isinstance(waveform_id, WaveformStreamID):
        return ''

    return waveform_id.get_seed_string()


def get_waveform_id(
    obj: Pick | Arrival | Amplitude | StationMagnitude,
    event: Event | None = None,
) -> WaveformStreamID | None:
    """Return waveform_id associated with ObsPy object."""
    if obj is None:
        return None
    if 'waveform_id' in obj and obj.waveform_id is not None:
        return obj.waveform_id

    pick = get_pick(obj, event)

    if not pick:
        return None

    return pick.waveform_id


def get_pick(
    obj: Pick | Arrival | Amplitude | StationMagnitude,
    event: Event | None = None,
) -> Pick | None:
    """
    Look up Pick associated with object.

    Event is not required for Arrival or Amplitude if referential integrity is
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
def get_amplitude(
    obj: Pick | Arrival | Amplitude | StationMagnitude,
    event: Event | None = None,
) -> Amplitude | None:
    """
    Look up Amplitude associated with object.

    Event is not required for Arrival or Amplitude if referential integrity is
    intact.
    """
    if obj is None:
        return None

    assert isinstance(obj, (Pick, Arrival, Amplitude, StationMagnitude))

    if isinstance(obj, Amplitude):
        return obj

    if isinstance(obj, Pick):
        if event is None:
            return None
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


class Tee(object):
    """
    Temporarily fork output to stdout and other files.

    Based on: http://stackoverflow.com/questions/11325019/
    """

    def __init__(self, *files):
        """Construct forker."""
        self.files = files

    def __del__(self):
        """Close upon deletion."""
        self.close()

    def open(self):
        """Redirect stdout."""
        if not hasattr(sys, '_stdout'):
            # Only do this once just in case stdout was already initialized
            # @note Will fail if stdout for some reason changes
            sys._stdout = sys.stdout  # pylint: disable=protected-access
        sys.stdout = self
        return self

    def close(self):
        """Restore normal operation."""
        stdout = sys._stdout  # pylint: disable=protected-access
        for file in self.files:
            if file != stdout:
                file.close()
        sys.stdout = stdout

    def write(self, obj):
        """Write to teed files."""
        for file in self.files:
            file.write(obj)

    def flush(self):
        """Flush teed files."""
        for file in self.files:
            file.flush()
