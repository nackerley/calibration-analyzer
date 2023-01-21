r"""
Fit transfer function estimate using Levenberg-Marquardt algorithm.

The variables to be fitted are the non-trivial normalized numerator and
denominator coefficients, organized in a single vector as follows:

$$x = [x_a x_b] = [a_1 ... a_m b_0 ... b_n]$$

After pre-computing all required powers of angular freqencies as
$\Omega$, and if $\div$ denotes elementwise division, the transfer
function is:

$$H = \Omega_b x_b \div (\Omega_m + \Omega_a x_a)$$

The function to be minimized, in matrix notation where $\times$ and
$\div$ denote elementwise multiplication and division, respectively,
$\sigma^2$ is the variance, and $\hat H$ and $H$ is the measured and
modeled response, respectively:

$$f(x) = \frac{1}{\hat \sigma \times |\hat H|} \times (H - \hat H)$$

Given coefficients x and transfer function estimate H:

$$J(x) = $$

@nackerle
"""
# TODO: give complete formula for Jacobian
import os
from typing import Optional, Sequence, Tuple
from logging import getLogger
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import ZerosPolesGain, TransferFunction
from scipy.optimize import least_squares, OptimizeResult

from calan.core import sort_complex, sensitivity, zpk_divide, zpk_multiply

np.random.seed(seed=42)


def zpk_out_of_band(
    system: ZerosPolesGain,
    f: ArrayLike,
    factor: float = 2,
    norm_freq_hz: float = 1,
    set_sensitivity: float = 1,
) -> ZerosPolesGain:
    """
    Construct out-of-band part of nominal response, with unity gain.

    Band is expanded by configurable factor beyond given frequency range.

    Tranfer function returned is constrained to have the sensitivity given at
    the normalization frequency.
    """
    f = np.array(f)
    f.sort()
    w_min = f[0]/factor
    w_max = f[-1]*factor
    poles = sort_complex(system.poles)
    zeros = sort_complex(system.zeros)

    if (np.abs(poles) < w_min).any():
        p_fixed, z_fixed = zip(*[
            (pole, zero) for pole, zero in zip(poles, zeros)
            if np.abs(pole) < w_min])
    else:
        p_fixed, z_fixed = [], []
    p_fixed += [pole for pole in poles if np.abs(pole) > w_max]
    z_fixed += [zero for zero in zeros if np.abs(zero) > w_max]

    return ZerosPolesGain(
        z_fixed, p_fixed, set_sensitivity /
        sensitivity(ZerosPolesGain(z_fixed, p_fixed, 1), norm_freq_hz))


WEIGHTING_SCHEMES = ['variance', 'response', 'frequency']


def fit_response(
    zpk_guess: ZerosPolesGain,
    f_meas: ArrayLike,
    h_meas: ArrayLike,
    var_meas: Optional[ArrayLike] = None,
    zpk_fixed: ZerosPolesGain = ZerosPolesGain([], [], 1),
    ftol: float = 1e-10,
    gtol: float = 1e-06,
    var_max: float = 0.1,
    weighting: Sequence[str] = ('variance', 'frequency'),
    debug: bool = False,
) -> Tuple[ZerosPolesGain, OptimizeResult]:
    """
    Obtain least-squares best-fit poles, zeros and gain using exact Jacobian.

    The Levenberg-Marquardt algorithm is used to obtain the weighted
    least-squares best fit, given the measured transfer function, expected
    variance in said estimates and initial guesses for poles, zeros and gain.

    Optional variance is typically computed from measured coherence and
    number of FFT windows used.

    Inputs:
      - zpk_nom:   initial guess for transfer function
      - f_meas:    measurement frequencies
      - h_meas:    estimated transfer function at frequencies
      - var_meas:  estimated variance at frequencies
      - zpk_fixed: fixed part of transfer function
      - gtol:      tolerance for termination by the norm of the gradient
      - var_max:   maximum variance for inclusion in fit
      - weighting: multple options can be selected
        - 'variance'    1/sqrt(var) to account for measurement errors
        - 'response'    1/abs(h_nom) to emphasize importance of passband
        - 'frequency'   1/f to account for over-weighting of high frequencies by FFT

    Outputs:
      - zpk_fit:   best_fit transfer function
      - result:    optimization result
    """
    logger = getLogger(__name__)
    if var_meas is None:
        var_meas = np.ones_like(f_meas)

    f_meas = np.array(f_meas).reshape(-1)
    h_meas = np.array(h_meas).reshape(-1)
    var_meas = np.array(var_meas).reshape(-1)
    if not f_meas.shape == h_meas.shape == var_meas.shape:
        raise ValueError(
            'Frequencies, transfer function estimates and variances '
            'must have same shape.')

    if not np.isrealobj(f_meas) or (f_meas < 0).any():
        raise TypeError('Frequencies must be real and positive')

    if not np.iscomplexobj(h_meas):
        raise TypeError('Transfer function estimates must be complex')

    if not np.isrealobj(var_meas) or (var_meas < 0).any():
        raise TypeError('Variances must be real and positive')

    if len(zpk_fixed.zeros) + len(zpk_fixed.poles):
        logger.info(
            'Fixing %d zeros and %d poles at nominal values.',
            len(zpk_fixed.zeros), len(zpk_fixed.poles))
    zpk_guess_unfixed = zpk_divide(zpk_guess, zpk_fixed)
    h_meas /= np.abs(zpk_fixed.freqresp(w=2*np.pi*f_meas)[1])
    tf_guess = zpk_guess_unfixed.to_tf()

    # extract coefficient vector
    a_guess = tf_guess.den
    b_guess = tf_guess.num
    if a_guess[0] == 0:
        logger.warning('Trimming leading zeroes from denominator coefficients.')
        a_guess = np.trim_zeros(a_guess, trim='f')
    if b_guess[0] == 0:
        logger.warning('Trimming leading zeroes from numerator coefficients.')
        b_guess = np.trim_zeros(b_guess, trim='f')

    p = np.argmax(np.flip(b_guess != 0))  # pylint: disable=invalid-name
    logger.info('Fixing %d zeros at zero.', p)
    b_guess = b_guess[:int(-p)]

    if a_guess[0] != 1:
        logger.warning('Normalizing denominator of initial guess.')
        b_guess = b_guess / a_guess[0]
        a_guess = a_guess / a_guess[0]

    # TODO: Determine whether denominator must be constrained to be stable.
    a_temp = apolystab(a_guess)
    if any(a_temp != a_guess):
        logger.warning('Stabilizing denominator of initial guess.')
        a_guess = np.copy(a_temp)

    # determine fitting orders
    n = b_guess.shape[0] + p - 1  # pylint: disable=invalid-name
    m = a_guess.shape[0] - 1  # pylint: disable=invalid-name
    N = f_meas.shape[0]  # pylint: disable=invalid-name

    if n > m:
        raise ValueError(
            f'Transfer function must be proper, but numerator degree {n} '
            f'is greater than denominator degree {m}.')
    if m + n - p > (var_meas < var_max).sum():
        raise ValueError(
            f'System must be overdetermined, but variance > {var_max} for '
            f'{(var_meas < var_max).sum()}/{len(var_meas)} measurements and '
            f'there are {m + n - p} coefficients to be fit.')

    # initial guess
    x_initial = np.hstack((a_guess[1:], b_guess))

    # precompute all required powers of frequencies
    omega = np.ones((N, m + 1), dtype=complex)
    w_meas = 2*np.pi*f_meas
    for i in range(m):
        omega[:, m - i - 1] = 1j*w_meas*omega[:, m - i]

    # subsets needed for computation of model and Jacobian
    omega_m = omega[:, 0]
    omega_a = omega[:, 1:m + 1]
    omega_b = omega[:, m - n:m - p + 1]

    def model(x):
        """Compute fitted transfer function model as function of frequency."""
        return omega_b @ x[m:] / (omega_m + omega_a @ x[:m])

    h_initial = model(x_initial)
    wt_meas = get_weights(weighting, f_meas, var_meas, h_initial)

    if debug:
        # pylint: disable=import-outside-toplevel
        import inspect
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        example_weightings = []
        for num in range(len(WEIGHTING_SCHEMES) + 1):
            example_weightings += list(combinations(WEIGHTING_SCHEMES, num))
        for ex_weighting in example_weightings:
            ex_weights = get_weights(ex_weighting, f_meas, var_meas, h_initial)
            ax.loglog(f_meas, ex_weights, label=','.join(ex_weighting) or 'none')
        ax.set_ylabel('Weight')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Frequency [Hz]')
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        weighting_png = os.path.splitext(module.__file__)[0] + '_weighting.png'  # type: ignore
        logger.info('Writing: %s', weighting_png)
        fig.savefig(weighting_png, bbox_inches='tight')

    def residuals(x):
        """Compute weighted mean squared error over all frequencies."""
        return wt_meas * (model(x) - h_meas)

    def real_residuals(x):
        """Convert residuals to vector of floats."""
        result = residuals(x)
        return np.hstack((result.real, result.imag))

    def jacobian(x):
        """Compute partial derivatives of residuals wrt coefficients."""
        model_den = omega_m + omega_a @ x[:m]
        part_a = -omega_a*(model(x) / model_den).reshape(-1, 1)
        part_b = omega_b / model_den.reshape(-1, 1)
        return np.hstack((part_a, part_b)) * wt_meas.reshape(-1, 1)

    def real_jacobian(x):
        """Convert Jacobian to vector of floats."""
        result = jacobian(x)
        return np.vstack((result.real, result.imag))

    result = least_squares(
        real_residuals, x_initial, jac=real_jacobian, method='lm',
        ftol=ftol, gtol=gtol, x_scale=1, verbose=1)

    logger.info(result.message)
    x_fit = result.x
    b_fit = np.hstack((x_fit[m:], np.zeros(p)))
    a_fit = np.hstack((1, x_fit[:m]))
    tf_fit = TransferFunction(b_fit, a_fit)
    zpk_fit = tf_fit.to_zpk()

    # restore fixed part
    zpk_fit = zpk_multiply(zpk_fit, zpk_fixed)

    return zpk_fit, result


def get_weights(weighting, f_meas, var_meas, h_initial):
    """Construct various kinds of weighting schemes."""
    unsupported = [item for item in weighting if item not in WEIGHTING_SCHEMES]
    if any(unsupported):
        raise ValueError(
            f'Weighting {unsupported} not in supported weighting schemes: '
            f'{WEIGHTING_SCHEMES}')

    wt_meas = np.ones_like(f_meas)
    if 'variance' in weighting:
        wt_meas /= np.sqrt(var_meas)
    if 'response' in weighting:
        wt_meas /= np.abs(h_initial)
    if 'frequency' in weighting:
        wt_meas /= np.sqrt(f_meas)
    wt_meas /= wt_meas.sum()
    return wt_meas


def apolystab(
    polynomial: np.ndarray,
) -> np.ndarray:
    """Return stabilized denominator polynomial of real analog filter."""
    if polynomial.ndim != 1:
        raise ValueError('Coefficients must be a vector.')
    if polynomial.shape[0] > 0:
        roots = np.roots(polynomial)
        real_positive = np.real(roots) > 0
        if real_positive.any():
            roots[real_positive] = -roots[real_positive]
            polynomial = np.real(np.poly(roots))
    return polynomial
