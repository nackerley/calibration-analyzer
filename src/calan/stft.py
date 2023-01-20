"""
Short-term Fourier Transform.

## References

Bendat, J. S., & Piersol, A. G. (2010). Random Data: Analysis and measurement
procedures (4th ed.). Wiley.
"""
# pylint: disable=consider-using-f-string
from __future__ import annotations

import os

import numpy as np
from numpy.typing import ArrayLike
from scipy import fftpack
from scipy.signal._spectral_py import _spectral_helper

from calan.core import get_logger
from calan.utilities import preferred_number

THIS_FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(THIS_FILE_NAME)[0] + '.log'


def num_windows_welch(len_signal, len_fft, len_overlap=None):
    """Return number of windows resulting from Welch's method."""
    if len_overlap is None:
        len_overlap = int(len_fft/2)

    return int((len_signal - len_fft)/(len_fft - len_overlap)) + 1


def len_fft_welch(len_signal, num_windows=None, fraction_overlap=None):
    """Recommended FFT length to achieve number of windows using Welch's method."""
    if num_windows is None:
        num_windows = 30
    if fraction_overlap is None:
        fraction_overlap = 0.5

    return int(preferred_number(
        len_signal / ((num_windows - 0.5) * (1 - fraction_overlap) + 1)))


def window_times_welch(len_signal, len_fft, f_sample, len_overlap=None):
    """
    Array of times of centers of windows resulting from Welch's method.

    Nearly verbatim from scipy.signal._spectral_helper().
    """
    if len_overlap is None:
        len_overlap = int(len_fft/2)

    return np.arange(len_fft/2, len_signal - len_fft/2 + 1,
                     len_fft - len_overlap)/f_sample


def fft_frequencies(len_fft, f_sample, sides='onesided'):
    """
    Array of frequencies expected from an FFT calculation.

    Nearly verbatim from scipy.signal._spectral_helper().
    """
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


class Stft():
    """Short-term fourier auto- and cross-spectra between input and output."""

    def __init__(self: Stft, log_level: str = 'INFO') -> None:
        """Construct STFT."""
        self.f = np.array([[np.NaN]])
        self.t = np.array([[np.NaN]])
        self.p_xx = np.array([[np.NaN]])
        self.p_yy = np.array([[np.NaN]])
        self.p_xy = np.array([[np.NaN]])
        self.logger = get_logger(self.__class__.__name__, LOG_FILE_NAME,
                                 log_level)

    def __str__(self: Stft) -> str:
        """Human-readable representation."""
        lines = [self.__class__.__name__ + ':']
        if np.isnan(self.p_xy).all():
            lines[0] = lines[0] + ' None'
        else:
            lines.append(
                f'    {self.p_xx.shape[0]} input x {self.p_yy.shape[0]} outputs ')
            lines.append(
                f'    t: {len(self.t)} from {self.t[0]} to {self.t[-1]} s')
            lines.append(
                f'    f: {len(self.f)} from {self.f[0]} to {self.f[-1]} Hz')
        return '\n'.join(lines)

    @staticmethod
    def mean(p_xy: ArrayLike) -> np.ndarray:
        """Finishing touch of Welch's method when deriving results."""
        p_xy = np.array(p_xy)
        if len(p_xy.shape) >= 2 and p_xy.size > 0:
            if p_xy.shape[-1] > 1:
                p_xy = np.mean(p_xy, axis=-1)
            else:
                p_xy = np.reshape(p_xy, p_xy.shape[:-1])
        return p_xy  # type: ignore

    def num_windows(self: Stft) -> int:
        """Return number of windows used."""
        return self.p_xx.shape[2]

    def compute(
        self: Stft,
        x: ArrayLike,
        y: ArrayLike,
        f_sample: float,
        len_fft: int,
        len_overlap: int,
        window: str = 'hann'
    ) -> None:
        """Detrend segments by removing constant value before windowing."""
        f_expected = fft_frequencies(len_fft, f_sample)
        x = np.array(x)
        y = np.array(y)
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))
        num_samples = x.shape[1]
        if y.shape[1] != num_samples:
            raise ValueError(
                'Signal length of output %d does not match input %d' %
                (y.shape[1], num_samples))
        num_windows = num_windows_welch(num_samples, len_fft, len_overlap)
        self.logger.info(
            '%d segments from %g to %g Hz',
            num_windows, f_expected[1], f_expected[-1])

        # pylint: disable=protected-access
        self.f, self.t, self.p_xy = _spectral_helper(
            x, y, window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')

        self.p_xx = _spectral_helper(
            x, x, window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]

        self.p_yy = _spectral_helper(
            y, y, window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]
        # pylint: enable=protected-access

    def trim(
        self: Stft,
        low_frequency_points: int = 1,
        high_frequency_fraction: float = 0.8,
    ) -> None:
        """
        Trim low- and high-frequency points.

        Typically low-frequency measurements are spoiled by imperfect DC
        removal. Similarly high-frequency measurements beyond the decimation
        filter corner are not useful.
        """
        keep = ((self.f >= self.f[low_frequency_points]) &
                (self.f <= self.f[-1]*high_frequency_fraction))
        self.f = self.f[keep]
        self.p_xx = self.p_xx[..., keep, :]
        self.p_yy = self.p_yy[..., keep, :]
        self.p_xy = self.p_xy[..., keep, :]

    def tf_estimate(self: Stft, alpha: int = 0) -> np.ndarray:
        """
        Return transfer function estimate (from input, x, to output, y).

        Optionally, differentiated alpha times to convert between displacement,
        acceleration and velocity.

        Bendat & Piersol (2014) Equation 9.53, p. 299.
        """
        return (self.mean(self.p_xy) /
                self.mean(self.p_xx))*(1j*2*np.pi*self.f)**alpha

    def coherence_squared(self: Stft) -> np.ndarray:
        """
        Return Welch's method squared coherence.


        Bendat & Piersol (2014) Equation 9.54, p. 299.
        """
        return np.abs(self.mean(self.p_xy))**2/(
            self.mean(self.p_xx)*self.mean(self.p_yy))

    def variance(self: Stft) -> np.ndarray:
        """
        Return Welch's method variance.

        Bendat & Piersol (2014) Table 9.6, p.312.
        """
        return (1/self.coherence_squared() - 1)/(2*len(self.t))
