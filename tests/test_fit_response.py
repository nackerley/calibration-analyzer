"""
Created on Thu Oct 31 17:08:44 2019

@author: nackerle
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import ArrayLike
from numpy.testing import assert_allclose
from scipy.signal import ZerosPolesGain, TransferFunction

from calan.core import (
    sensitivity, lti_multiply, lti_divide, gain_db, phase_deg)
from calan.fit_response import (
    precompute_omega, model, residuals, jacobian,
    fit_response, zpk_out_of_band, get_weights)

np.set_printoptions(
    suppress=False, precision=3, linewidth=200, floatmode='maxprec')

# nominal response is that of Trillium 120QA
ZPK_NOM = ZerosPolesGain(
    [0, -31.63,
     -350],
    [-0.036614-0.037059j, -0.036614+0.037059j, -32.55, -142,
     -364-404j, -364+404j, -1260, -4900-5200j, -4900+5200j,
     -7100-1700j, -7100+1700j],
    5.2018e+24)
F_MEAS = np.arange(0.004, 40.002, 0.004)
# nominal variance is reflection of typical ambient noise
FREQ_VARIANCE = [
    (0.001, 1e-1),
    (1, 1e-7),
    (5, 5e-8),
    (100, 1e-4),
]
GUESS_SD = 0.1
DEBUG = True

np.random.seed(42)


def _complex_errors(values: ArrayLike, stdev: ArrayLike) -> np.ndarray:
    """Apply normally distributed errors to complex values."""
    values = np.array(values)
    errors = np.random.normal(
        loc=0, scale=np.array(stdev).reshape(-1, 1), size=(len(values), 2)
    ).view(np.complex128).squeeze()*abs(values)
    new_values = values + errors
    return new_values


def _feature_errors(values: ArrayLike, stdev: ArrayLike) -> np.ndarray:
    """Apply errors, retaining complex pairs and real vs. complex."""
    values = np.array(values)
    stdev = np.array(stdev)
    uniques = np.array(
        [value for i, value in enumerate(values)
         if i == 0 or value == 0 or value != np.conj(values[i - 1])]
    ).reshape(-1, 1)
    uniques_2d = uniques.view(np.float64)
    errors_2d = np.random.normal(
        loc=0, scale=stdev, size=uniques_2d.shape
    )*np.tile(np.abs(uniques), (1, 2))
    errors_2d[uniques_2d == 0] = 0
    new_uniques = (uniques_2d + errors_2d).view(np.complex128).reshape(-1)

    new_values = []
    for value in new_uniques:
        if np.iscomplex(value):
            new_values += [value, np.conj(value)]
        else:
            new_values += [value]
    return np.array(new_values)


def test_jacobian() -> None:
    """
    Compare jacobian to analytic result and 2-point approximation.

    Dummy model is second-order high-pass cascaded with a lead-lag, to make it
    3rd order in numerator and denominator, with 2 zeros at zero.
        ((s + wz1)*s**2)/((s + wp1)*(s**2 + 2*dp2*wp2*s + wp2**2))
    """
    # set up standard model
    k = 2
    wz1 = 2*np.pi*10
    wp1 = 2*np.pi*20
    wp2 = 2*np.pi*0.1
    dp2 = 1/np.sqrt(2)
    a = [1,  # pylint: disable=invalid-name
         2*dp2*wp2 + wp1,
         wp2**2 + 2*dp2*wp2*wp1,
         wp1*wp2**2]
    b = [k,  # pylint: disable=invalid-name
         k*wz1,
         0,
         0]
    tf_ = TransferFunction(b, a)
    f = np.logspace(-2, 2, num=9)
    s = 2j*np.pi*f  # pylint: disable=invalid-name
    h_model = tf_.freqresp(2*np.pi*f)[1]
    weights = 1/np.abs(h_model)

    # re-express in terms of our model
    p = int(np.argmax(np.flip(tf_.num != 0)))
    m = len(tf_.den) - 1
    n = len(tf_.num) - 1
    x = np.hstack((tf_.den[1:], tf_.num[:n - p + 1]))

    analytic = (weights * np.vstack([
        -(b[0]*s**3 + b[1]*s**2)*s**2 / (s**3 + a[1]*s**2 + a[2]*s + a[3])**2,
        -(b[0]*s**3 + b[1]*s**2)*s / (s**3 + a[1]*s**2 + a[2]*s + a[3])**2,
        -(b[0]*s**3 + b[1]*s**2) / (s**3 + a[1]*s**2 + a[2]*s + a[3])**2,
        s**3 / (s**3 + a[1]*s**2 + a[2]*s + a[3]),
        s**2 / (s**3 + a[1]*s**2 + a[2]*s + a[3])])).T
    approximate = np.zeros_like(analytic)
    for i, _ in enumerate(x):
        d_x = x.copy()
        d_x[i] *= 1 + 1e-6
        d_model = TransferFunction(
            np.hstack((d_x[m:], np.zeros(p))),
            np.hstack((1, d_x[:m]))).freqresp(2*np.pi*f)[1]
        approximate[:, i] = weights*(d_model - h_model)/(d_x[i] - x[i])

    omega = precompute_omega(f, m)
    # weights is only being passed here as a dummy to satisfy mypy
    matrix = jacobian(x, omega, weights, weights, m, n, p)

    assert_allclose(matrix, approximate, rtol=1e-3)
    assert_allclose(analytic, approximate, rtol=1e-3)
    assert_allclose(matrix, analytic)


def test_model_residual() -> None:
    """Verify that matrix-based models and residuals are correct."""
    direct_model = ZPK_NOM.freqresp(2*np.pi*F_MEAS)[1]

    tf_nom = ZPK_NOM.to_tf()
    p = int(np.argmax(np.flip(tf_nom.num != 0)))
    a_nom = tf_nom.den
    b_nom = tf_nom.num[:-p]
    n = len(b_nom) + p - 1
    m = len(a_nom) - 1
    x_nom = np.hstack((a_nom[1:], b_nom))

    omega = precompute_omega(F_MEAS, m)
    matrix_model = model(x_nom, omega, m, n, p)

    assert_allclose(direct_model, matrix_model, rtol=1e-12)

    var_meas = 10**(
        np.interp(np.log10(F_MEAS), *list(zip(*np.log10(FREQ_VARIANCE)))))
    synthetic_data = _complex_errors(direct_model, var_meas)
    weights = get_weights(('response', 'variance'), F_MEAS, var_meas,
                          direct_model, (1e-6, 0.1))
    direct_res = weights*(direct_model - synthetic_data)

    matrix_res = residuals(x_nom, omega, synthetic_data, weights, m, n, p)

    assert_allclose(direct_res, matrix_res, rtol=1e-6)


def test_fit_response() -> None:
    """Verify fitting with synthetic data and a bad initial guess."""

    nominal_data = ZPK_NOM.freqresp(2*np.pi*F_MEAS)[1]

    var_meas = 10**(
        np.interp(np.log10(F_MEAS), *list(zip(*np.log10(FREQ_VARIANCE)))))
    synthetic_data = _complex_errors(nominal_data, var_meas)

    zpk_fixed = zpk_out_of_band(ZPK_NOM, F_MEAS, norm_freq_hz=1)
    zpk_nom_unfixed = lti_divide(ZPK_NOM, zpk_fixed)

    poles_error = _feature_errors(zpk_nom_unfixed.poles, GUESS_SD)
    zeros_error = _feature_errors(zpk_nom_unfixed.zeros, GUESS_SD)
    sensitivity_error = (np.random.normal(loc=1, scale=GUESS_SD) *
                         sensitivity(zpk_nom_unfixed))

    zpk_error = ZerosPolesGain(
        zeros_error, poles_error, sensitivity_error /
        sensitivity(ZerosPolesGain(zeros_error, poles_error, 1)))
    zpk_guess = lti_multiply(zpk_fixed, zpk_error)
    guess_data = zpk_guess.freqresp(2*np.pi*F_MEAS)[1]

    os.chdir(Path(__file__).parent)
    zpk_fit = fit_response(  # pylint: disable=unused-variable
        zpk_guess, F_MEAS, synthetic_data, var_meas, zpk_fixed, debug=DEBUG)
    zpk_fit_unfixed = lti_divide(zpk_fit, zpk_fixed)
    fit_data = zpk_fit.freqresp(2*np.pi*F_MEAS)[1]

    if DEBUG:
        _plot_responses(F_MEAS, nominal_data, synthetic_data, guess_data,
                        fit_data, var_meas)

    assert_allclose(sensitivity(zpk_fit), sensitivity(ZPK_NOM), rtol=1e-4)
    assert_allclose(zpk_fit_unfixed.poles, zpk_nom_unfixed.poles, rtol=1e-2)
    assert_allclose(zpk_fit_unfixed.zeros, zpk_nom_unfixed.zeros, rtol=1e-2)


def _plot_responses(f, nominal_data, synthetic_data, guess_data, fit_data,
                    var_meas):
    """Plot nominal and fit responses.

    Results are differentiated to make them plot flatter,
    because typically output is velocity and input is acceleration.
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.5, 8.5))
    fig.subplots_adjust(hspace=0)
    axes[0].semilogx(f, gain_db(guess_data) - gain_db(nominal_data),
                     label='guess')
    axes[0].semilogx(f, gain_db(synthetic_data) - gain_db(nominal_data),
                     label='synthetic')
    axes[0].semilogx(f, gain_db(fit_data) - gain_db(nominal_data),
                     label='fit')
    axes[0].set_ylabel('Gain wrt nominal [dB]')
    axes[0].legend()

    axes[1].semilogx(f, phase_deg(guess_data) - phase_deg(nominal_data),
                     label='guess')
    axes[1].semilogx(f, phase_deg(synthetic_data) - phase_deg(nominal_data),
                     label='synthetic')
    axes[1].semilogx(f, phase_deg(fit_data) - phase_deg(nominal_data),
                     label='fit')
    axes[1].set_ylabel('Phase wrt nominal [°]')

    axes[2].loglog(f, var_meas, 'k', label='variance')
    axes[2].legend()
    axes[2].set_ylabel('Variance')
    axes[2].set_xlabel('Frequency [Hz]')
    fig.savefig(Path(__file__).stem + '.png', bbox_inches='tight')
