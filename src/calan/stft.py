# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:28:06 2020

@author: nackerle
"""
import os
import numpy as np
from scipy import fftpack
import scipy.signal as sp

from calan.core import get_logger
from calan.utilities import preferred_number

THIS_FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(THIS_FILE_NAME)[0] + '.log'


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


class Stft():
    '''
    Short-term fourier auto- and cross-spectra between input (x) and output
    (y) signals.
    '''

    def __init__(self, f=None, t=None, p_xx=None, p_yy=None, p_xy=None,
                 log_level='INFO'):

        self.f = f
        self.t = t
        self.p_xx = p_xx
        self.p_yy = p_yy
        self.p_xy = p_xy
        self.logger = get_logger(self.__class__.__name__, LOG_FILE_NAME,
                                 log_level)

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        if self.p_xy is None:
            lines[0] = lines[0] + ' None'
        else:
            lines.append('\tt: %d from %g to %g s' %
                         (len(self.t), self.t[0], self.t[-1]))
            lines.append('\tf  %d from %g to %g Hz' %
                         (len(self.f), self.f[0], self.f[-1]))
        return '\n'.join(lines)

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
        f_expected = fft_frequencies(len_fft, f_sample)
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
            y, y, window=window,
            fs=f_sample, nperseg=len_fft, noverlap=len_overlap, mode='psd')[2]
        # pylint: enable=protected-access

    def trim(self, low_frequency_points=1, high_frequency_fraction=0.8):
        '''
        Trim low- and high-frequency points.

        Typically low-frequency measurements are spoiled by imperfect DC
        removal. Similarly high-frequency measurements beyond the decimation
        filter corner are not useful.
        '''
        keep = ((self.f >= self.f[low_frequency_points]) &
                (self.f <= self.f[-1]*high_frequency_fraction))
        self.f = self.f[keep]
        self.p_xx = self.p_xx[..., keep, :]
        self.p_yy = self.p_yy[..., keep, :]
        self.p_xy = self.p_xy[..., keep, :]

    def coherence_squared(self):
        '''
        Return Welch's method squared coherence.
        '''
        return np.abs(self._mean(self.p_xy))**2/(
            self._mean(self.p_xx)*self._mean(self.p_yy))

    def variance(self):
        '''
        Return Welch's method variance.
        '''
        return (1/self.coherence_squared() - 1)/(2*len(self.t))

    def tf_estimate(self, alpha=0):
        '''
        Return transfer function estimate (from input, x, to output, y),
        differentiated alpha times.
        '''
        return (self._mean(self.p_xy) /
                self._mean(self.p_xx))*(1j*2*np.pi*self.f)**alpha
