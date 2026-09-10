"""A collection of utilities for transfer function fitting."""
# SPDX-FileCopyrightText: 2026 His Majesty the Κing in Right of Canada <copyright.droitdauteur@pch.gc.ca>  # noqa: E501
#
# SPDX-License-Identifier: GPL-3.0-or-later

# pylint: disable=too-many-lines
import logging
from copy import deepcopy
from logging import getLogger, Logger
from logging.config import dictConfig
from typing import Type

import numpy as np
from numpy.typing import NDArray
import scipy.signal as sp
from scipy.signal import lti, ZerosPolesGain, TransferFunction, StateSpace
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from obspy import Stream
from obspy.core.inventory import (
    ResponseStage, InstrumentSensitivity, PolesZerosResponseStage,
    CoefficientsTypeResponseStage, FIRResponseStage)
from obspy.core.event import Pick, Arrival, Amplitude, StationMagnitude, Event

from calan import PACKAGE_VERSION


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
    if isinstance(to_type, ZerosPolesGain):
        return system.to_zpk()
    if isinstance(to_type, TransferFunction):
        return system.to_tf()
    if isinstance(to_type, StateSpace):
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

    LOG_SETTINGS['handlers']['console'].update(  # type: ignore
        {'level': log_console_level})
    LOG_SETTINGS['handlers']['file'].update(  # type: ignore
        {'filename': log_file_name})
    dictConfig(LOG_SETTINGS)

    logger = logging.getLogger(name)

    logger.info('%s: %s', log_file_name, PACKAGE_VERSION)

    return logger


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
