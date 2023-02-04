"""A collection of functions useful for seismograph calibration."""
# pylint: disable=consider-using-f-string
import os
from warnings import warn
from struct import calcsize
from logging import getLogger
from typing import Callable, List, Optional, Sequence, Sized, Tuple
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import ArrayLike

from calan.utilities import parse_duration, parse_voltage, pretty_bytes
from calan.core import compute_decim_delay

# Nanometrics Centaur User Guide 17935R5, 2016-11-02
CALIBRATION_SAMPLE_RATE = 30e3
SAMPLE_FORMAT = '<h'  # little-endian short (16-bit) integer


def generate_piecewise_constant(
    durations: ArrayLike,
    voltages: ArrayLike,
) -> Tuple[np.ndarray, str]:
    """Generate a piecewise constant calibration signal in volts."""
    logger = getLogger(__name__)
    durations = np.asarray(durations)
    voltages = np.asarray(voltages)

    file_name = 'step_%s' % '_'.join(
        ['%gV_%ss' % (voltage, duration)
         for voltage, duration in zip(voltages, durations)])
    file_size = expected_wav_size(durations.sum())
    logger.debug('Uncompressed output "%s.wav" will be %s.',
                 file_name, file_size)

    times = durations.cumsum()
    t = np.arange(0, durations.sum(), 1/CALIBRATION_SAMPLE_RATE).reshape(-1, 1)
    indices = np.argmax(t < times, axis=1)

    return voltages[indices], file_name


def expected_wav_size(duration: float) -> str:
    """Compute expected size before compression."""
    return pretty_bytes(
        calcsize(SAMPLE_FORMAT)*duration*CALIBRATION_SAMPLE_RATE)


def parse_signal_file_name(
    file_name: str,
) -> Tuple[str, float, float, float, float, float, float, float]:
    """
    Parse a calibration signal file name.

    Returns:
        style, duration_seconds, pp_voltage, rms_voltage, mean_voltage,
        lead_in, lead_out, sample_rate
    """
    file_name = file_name.replace('.wav', '').replace('.gz', '')
    parts = os.path.split(file_name)[1].split('_')

    if not any(char.isdigit() for char in parts[0]):
        style = parts.pop(0)
    else:  # for backward compatibility
        style = ''
    duration_seconds = float(parse_duration(parts.pop(1)))

    extras = []
    tokens: List[str] = ['pp', 'rms', 'mean', 'on', 'off', 'sps']
    parsers: List[Callable] = [
        parse_voltage, parse_voltage, parse_voltage, parse_duration,
        parse_duration, float]
    defaults: List[float] = [0, 0, 0, 0, 0, CALIBRATION_SAMPLE_RATE]
    for token, parser, default in zip(tokens, parsers, defaults):
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


def get_times(
    signal: ArrayLike,
    b_stages: Optional[Sequence[Sized]] = None,
    factors: Optional[Sequence[int]] = None,
    discard_initial: bool = True,
) -> np.ndarray:
    """Generate an array of times corresponding to a calibration signal."""
    signal = np.array(signal)
    i = np.arange(len(signal))
    if factors:
        i *= np.prod(factors)
    if factors and b_stages:
        if discard_initial:
            i += compute_decim_delay(b_stages, factors) // 2
        else:
            i -= compute_decim_delay(b_stages, factors) // 2
    return i/CALIBRATION_SAMPLE_RATE


def pad_for_decimation(
    signal: ArrayLike,
    b_stages: Sequence[Sized],
    factors: Sequence[int],
) -> Tuple[np.ndarray, float]:
    """
    Pad a signal with zeros in preparation for decimation.

    First sample after decimation will be at the same time as the first
    sample before decimation.
    """
    n_pad_upsample = compute_decim_delay(b_stages, factors)
    signal_padded = np.hstack((np.zeros((int(n_pad_upsample/2), )), signal,
                               np.zeros((int(n_pad_upsample/2), ))))
    t_start = -n_pad_upsample/2/CALIBRATION_SAMPLE_RATE
    return signal_padded, t_start


def sample_hold_digitize(signal: ArrayLike) -> np.ndarray:
    """Process signal as if sampled and held by DAC, then digitized by ADC."""
    signal = np.concatenate((np.zeros((1, )), signal, np.zeros((1, ))))
    signal = signal[:-1]/2 + signal[1:]/2
    return signal


def plot_calibration(
    signal: ArrayLike,
    units: str = 'counts',
    file_name: Optional[str] = None,
    sample_rate: float = 100,
) -> None:
    """
    Plot a calibration signal, decimated to given sample rate.

    CAUTION: decimation is performed without prior filtering, so resulting
    signal may be strongly aliased.
    """
    signal = np.array(signal)
    t = np.arange(len(signal))/CALIBRATION_SAMPLE_RATE

    step = int(CALIBRATION_SAMPLE_RATE/sample_rate)
    _, ax = plt.subplots()
    ax.plot(t[::step], signal[::step], marker='x')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal [%s]' % units)

    if file_name is not None:
        file_name = os.path.splitext(file_name)[0] + '.png'
        plt.gcf().savefig(file_name, dpi=300, bbox_inches='tight')


def plot_calibration_decimated(
    sig_in: ArrayLike,
    sig_out: ArrayLike,
    b_stages: Sequence[Sized],
    factors: Sequence[int],
    discard_initial: bool,
) -> None:
    """Compare timing of signals before/after decimation."""
    t_in = get_times(sig_in)
    t_out = get_times(sig_out, b_stages=b_stages, factors=factors,
                      discard_initial=discard_initial)

    _, ax = plt.subplots()
    ax.plot(t_in, sig_in, label='input')
    ax.plot(t_out, sig_out, marker='x', label='output')
    ax.set_xlabel('Time [s]')
    ax.grid()
    ax.legend()
