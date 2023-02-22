"""
Created on Sat Oct  8 04:49:16 2016

@author: nackerle
"""
import os
import numpy as np

from calan import DATA_PATH
from calan.core import (
    dataless2inventory, extract_decimation_coefficients, multi_decim)
from calan.calibration_toolbox import (
    generate_piecewise_constant, pad_for_decimation, get_times,
    sample_hold_digitize)


DURATIONS = [5, 5, 5]
VOLTAGES = [0, 4, 0]

inv = dataless2inventory(os.path.join(DATA_PATH, 'Centaur-T120Q.dataless'))
channel = inv.networks[0].stations[0].channels[0]
stages = channel.response.response_stages


def test_zero_crossings():
    """
    Verify that zero crossings in a piecewise constant signal stay put.
    """
    transitions = np.array(VOLTAGES[:-1])/2 + np.array(VOLTAGES[1:])/2

    sig_in, _ = generate_piecewise_constant(DURATIONS, VOLTAGES)
    sig_in = sample_hold_digitize(sig_in)
    t_in = get_times(sig_in)
    transitions_in = [sig_in[np.argmin(np.abs(t_in - time))]
                      for time in np.cumsum(DURATIONS)[:-1]]
    np.testing.assert_allclose(transitions_in, transitions)

    b_stages, factors = extract_decimation_coefficients(stages)
    signal_padded, t_start = pad_for_decimation(sig_in, b_stages, factors)

    sig_out = multi_decim(signal_padded, b_stages, factors)[0]
    t_out = t_start + get_times(sig_out, b_stages, factors)
    transitions_out = [sig_out[np.argmin(np.abs(t_out - time))]
                       for time in np.cumsum(DURATIONS)[:-1]]
    np.testing.assert_allclose(transitions_out, transitions, atol=1e-5)


def test_chunked_decimation():
    """
    Check input/output delays and leftover/unused samples.
    """
    signal_volts, _ = generate_piecewise_constant(DURATIONS, VOLTAGES)

    b_stages, factors = extract_decimation_coefficients(stages)
    sig_whole, _, sig_unused_whole = multi_decim(
        signal_volts, b_stages, factors)

    i_break = int(len(signal_volts)/2)
    chunk_1, z_mid, sig_unused_1 = multi_decim(
        signal_volts[:i_break], b_stages, factors)
    chunk_2, _, sig_unused_2 = multi_decim(
        signal_volts[i_break:], b_stages, factors,
        z_in=z_mid, sig_leftover=sig_unused_1,
        discard_initial=False)
    sig_chunked = np.hstack((chunk_1, chunk_2))

    np.testing.assert_allclose(sig_whole, sig_chunked)
    np.testing.assert_allclose(sig_unused_whole, sig_unused_2)
