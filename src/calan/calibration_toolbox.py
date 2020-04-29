# -*- coding: utf-8 -*-
"""
A collection of functions useful for seismograph calibration.
"""
# pylint: disable=logging-not-lazy
from __future__ import absolute_import, division, print_function

import os
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt

from calan.utilities import parse_duration, parse_voltage
from calan.core import compute_decim_delay

# %% constants

# Nanometrics Centaur User Guide 17935R5, 2016-11-02
CALIBRATION_SAMPLE_RATE = 30e3


# %% definitions
def parse_signal_file_name(file_name):
    '''
    Parse a calibration signal file name.

    :rtype: tuple
    :returns: style, duration_seconds, pp_voltage, rms_voltage, mean_voltage,
            lead_in, lead_out, sample_rate
    '''

    file_name = file_name.replace('.wav', '').replace('.gz', '')
    parts = os.path.split(file_name)[1].split('_')

    if not any(char.isdigit() for char in parts[0]):
        style = parts.pop(0)
    else:  # for backward compatibility
        style = ''
    duration_seconds = float(parse_duration(parts.pop(1)))

    extras = []
    for token, parser, default in zip(
            ['pp', 'rms', 'mean', 'on', 'off', 'sps'],
            [parse_voltage, parse_voltage, parse_voltage,
             parse_duration, parse_duration, float],
            [0, 0, 0, 0, 0, CALIBRATION_SAMPLE_RATE]):
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
    signal_padded = np.hstack((np.zeros((int(n_pad_upsample/2), )), signal,
                               np.zeros((int(n_pad_upsample/2), ))))
    t_start = -n_pad_upsample/2/CALIBRATION_SAMPLE_RATE
    return signal_padded, t_start


def sample_hold_digitize(signal):
    '''
    Process signal as if sampled and held by a DAC, then digitized by an ADC.
    '''
    signal = np.concatenate((np.zeros((1, )), signal, np.zeros((1, ))))
    signal = signal[:-1]/2 + signal[1:]/2
    return signal


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
        plt.gcf().savefig(file_name, dpi=300, bbox_inches='tight')


def plot_calibration_decimated(sig_in, sig_out, b_stages, factors,
                               discard_initial):
    '''
    Utility for comparing timing of signals before/after decimation.
    '''

    t_in = get_times(sig_in)
    t_out = get_times(sig_out, b_stages=b_stages, factors=factors,
                      discard_initial=discard_initial)

    _, ax = plt.subplots()
    ax.plot(t_in, sig_in, label='input')
    ax.plot(t_out, sig_out, marker='x', label='output')
    ax.set_xlabel('Time [s]')
    ax.grid()
    ax.legend()
