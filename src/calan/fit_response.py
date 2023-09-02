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
$\sigma^2$ is the variance, $w$ are some weights, and $\hat H$ and $H$ are
the measured and modeled response, respectively, is:

$$f(x) = w \times (H - \hat H)$$

And the associated Jacobian is:

$$J(x) = w \times \left[
    -\Omega_a \times \left( \Omega_b x_b \div (\Omega_m + \Omega_a x_a)^2
    \right) \Omega_b \div (\Omega_m + \Omega_a x_a ) \right]$$

Where the weighting can consist of one or all of inverse 1) square root of
variance, 2) response or 3) frequency (below all are shown):

$$w = \frac{1}{\hat \sigma \times |\hat H| \times f} $$

@nackerle
"""
from pathlib import Path
from math import log10
from io import StringIO
from contextlib import redirect_stdout
from typing import Optional, Sequence, Tuple
from logging import getLogger
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import lti, ZerosPolesGain, TransferFunction
from scipy.optimize import least_squares

from calan.core import (
    sort_complex, sensitivity, phase_deg,
    lti_minreal, lti_divide, lti_multiply, lti_is_proper)

np.random.seed(seed=42)


def extract_coefficients(system: lti) -> Tuple[np.ndarray, int, int, int]:
    """
    Extract coefficient vector from transfer function.

    Returns:
        - x: coefficient vector
        - p: number of poles fixed at zero
        - n: numerator degree
        - m: denominator degree
    """
    if isinstance(system, TransferFunction):
        tf_ = system
    else:
        tf_ = system.to_tf()

    logger = getLogger(__name__)
    den = tf_.den
    num = tf_.num
    if den[0] == 0:
        logger.warning(
            'Trimming leading zeroes from denominator coefficients.')
        den = np.trim_zeros(den, trim='f')
    if num[0] == 0:
        logger.warning('Trimming leading zeroes from numerator coefficients.')
        num = np.trim_zeros(num, trim='f')

    p = int(np.argmax(np.flip(num != 0)))
    logger.info('Fixing %d zeros at zero.', p)

    if den[0] != 1:
        logger.warning('Normalizing denominator.')
        num = num / den[0]
        den = den / den[0]

    a_temp = apolystab(den)
    if any(a_temp != den):
        logger.warning('Stabilizing denominator.')
        den = np.copy(a_temp)

    # determine fitting orders
    n = num.shape[0] - 1
    m = den.shape[0] - 1

    x = np.hstack((den[1:], num[:-p]))
    return x, p, n, m


def precompute_omega(f: ArrayLike, m: int) -> np.ndarray:
    """Precompute required powers of angular frequencies."""
    f = np.array(f)
    omega = np.ones((f.shape[0], m + 1), dtype=complex)
    w_meas = 2*np.pi*f
    for i in range(m):
        omega[:, m - i - 1] = 1j*w_meas*omega[:, m - i]
    return omega


def _get_views(
    x: ArrayLike, omega: ArrayLike,
    m: int, n: int, p: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get subsets of matrices needed for computations."""
    x = np.array(x)
    omega = np.array(omega)
    x_a = x[:m]
    x_b = x[m:]
    omega_a = omega[:, 1:m + 1]
    omega_b = omega[:, m - n:m - p + 1]
    omega_m = omega[:, 0]
    return x_a, x_b, omega_a, omega_b, omega_m


def model(
    x: ArrayLike, omega: ArrayLike,
    m: int, n: int, p: int,
) -> np.ndarray:
    """Compute fitted transfer function model as function of frequency."""
    x_a, x_b, omega_a, omega_b, omega_m = _get_views(x, omega, m, n, p)

    model_num = omega_b @ x_b
    model_den = omega_m + omega_a @ x_a
    return model_num / model_den


def residuals(
    x: ArrayLike, omega: ArrayLike, h_meas: ArrayLike, weights: ArrayLike,
    m: int, n: int, p: int,
) -> np.ndarray:
    """Compute weighted mean squared error over all frequencies."""
    x = np.array(x)
    omega = np.array(omega)
    h_meas = np.array(h_meas)
    weights = np.array(weights)

    return weights * (model(x, omega, m, n, p) - h_meas)


def real_residuals(
    x: ArrayLike, omega: ArrayLike, h_meas: ArrayLike, weights: ArrayLike,
    m: int, n: int, p: int,
) -> np.ndarray:
    """Convert residuals to vector of floats (wrapper for least_squares)."""
    return residuals(x, omega, h_meas, weights, m, n, p).view(float)


def jacobian(
    x: ArrayLike, omega: ArrayLike,
    h_meas: ArrayLike,  # pylint: disable=unused-argument
    weights: ArrayLike,
    m: int, n: int, p: int,
) -> np.ndarray:
    """Compute partial derivatives of residuals wrt coefficients."""
    x_a, x_b, omega_a, omega_b, omega_m = _get_views(x, omega, m, n, p)

    h_meas = np.array(h_meas)
    weights = np.array(weights)

    model_num = omega_b @ x_b
    model_den = np.array(omega_m + omega_a @ x_a)
    part_a = -omega_a*(model_num / model_den**2).reshape(-1, 1)
    part_b = omega_b / model_den.reshape(-1, 1)
    return np.hstack((part_a, part_b))*weights.reshape(-1, 1)


def real_jacobian(
    x: ArrayLike, omega: ArrayLike,
    h_meas: ArrayLike,  # pylint: disable=unused-argument
    weights: ArrayLike,
    m: int, n: int, p: int,
) -> np.ndarray:
    """Convert Jacobian to vector of floats (wrapper for least_squares)."""
    result = jacobian(x, omega, h_meas, weights, m, n, p)
    return np.vstack((result.real, result.imag))


WEIGHTING_SCHEMES = ['variance', 'response', 'frequency']
LEAST_SQUARES_METHODS = ['scipy.least_squares', 'line_search']


def fit_response(
    zpk_guess: ZerosPolesGain,
    f: ArrayLike,
    h_meas: ArrayLike,
    var_meas: Optional[ArrayLike] = None,
    zpk_fixed: ZerosPolesGain = ZerosPolesGain([], [], 1),
    ftol: float = 1e-10,
    gtol: float = 1e-06,
    var_lims: Tuple[float, float] = (1e-4, 0.1),
    weighting: Sequence[str] = ('variance', 'response', 'frequency'),
    method: str = 'line_search',
    debug: bool = False,
) -> ZerosPolesGain:
    """
    Obtain least-squares best-fit poles, zeros and gain using exact Jacobian.

    The Levenberg-Marquardt algorithm is used to obtain the weighted
    least-squares best fit, given the measured transfer function, expected
    variance in said estimates and initial guesses for poles, zeros and gain.

    Optional variance is typically computed from measured coherence and
    number of FFT windows used.

    Inputs:
      - zpk_guess:  initial guess for transfer function
      - f:          measurement frequencies
      - h_meas:     estimated transfer function at frequencies
      - var_meas:   estimated variance at frequencies
      - zpk_fixed:  fixed part of transfer function
      - gtol:       tolerance for termination by the norm of the gradient
      - var_lims:   minimum variance (weighting), maximum variance (inclusion)
      - weighting:  multple options can be selected
        - 'variance'    1/sqrt(var) to account for measurement errors
        - 'response'    1/abs(h_nom) to ensure residual matters across band
        - 'frequency'   1/f to balance over-weighting of high freqs. by FFT

    Outputs:
      - zpk_fit:   best_fit transfer function
      - result:    optimization result
    """
    logger = getLogger(__name__)
    if method not in LEAST_SQUARES_METHODS:
        raise RuntimeError(
            f"Method '{method}' not among supported: " +
            ', '.join(LEAST_SQUARES_METHODS))
    if var_meas is None:
        var_meas = np.ones_like(f)

    f = np.array(f).reshape(-1)
    h_meas = np.array(h_meas).reshape(-1)
    var_meas = np.array(var_meas).reshape(-1)
    if not f.shape == h_meas.shape == var_meas.shape:
        raise ValueError(
            'Frequencies, transfer function estimates and variances '
            'must have same shape.')
    if not np.isrealobj(f) or (f < 0).any():
        raise TypeError('Frequencies must be real and positive')
    if not np.iscomplexobj(h_meas):
        raise TypeError('Transfer function estimates must be complex')
    if not np.isrealobj(var_meas) or (var_meas < 0).any():
        raise TypeError('Variances must be real and positive')

    if len(zpk_fixed.zeros) + len(zpk_fixed.poles):
        logger.info(
            'Fixing %d zeros and %d poles at nominal values.',
            len(zpk_fixed.zeros), len(zpk_fixed.poles))
    zpk_guess_unfixed = lti_divide(zpk_guess, zpk_fixed)
    h_meas_unfixed = h_meas / zpk_fixed.freqresp(w=2*np.pi*f)[1]
    x_initial, p, n, m = extract_coefficients(zpk_guess_unfixed)

    if n > m:
        raise ValueError(
            f'Transfer function must be proper, but numerator degree {n} '
            f'is greater than denominator degree {m}.')
    if m + n - p > (var_meas < var_lims[1]).sum():
        raise ValueError(
            f'System must be overdetermined, but variance > {var_lims[1]} for '
            f'{(var_meas < var_lims).sum()}/{len(var_meas)} measurements and '
            f'there are {m + n - p} coefficients to be fit.')
    logger.info(
        'Fitting numerator/denominator degrees %d/%d at %d frequencies.',
        n, m, f.shape[0])

    omega = precompute_omega(f, m)
    h_initial = model(x_initial, omega, m, n, p)
    weights = get_weights(weighting, f, var_meas, h_initial, var_lims)

    if debug:
        _plot_possible_weights(
            weighting, f, var_meas, h_meas_unfixed, var_lims)

    model_parameters = dict(
        omega=omega, h_meas=h_meas_unfixed, weights=weights, m=m, n=n, p=p)
    logger.info('Method: %s', method)
    if method == 'line_search':
        x_fits, e_fits, message = least_squares_line_search(
            x_initial, **model_parameters)
        logger.info('Termination: %s', message)
        logger.info('Cost reduced from %.2g to %.2g in %d iterations.',
                    e_fits[0], e_fits[-1], len(e_fits))
        x_fit = x_fits[-1]
    else:
        captured_stdout = StringIO()
        with redirect_stdout(captured_stdout):
            result = least_squares(
                real_residuals, x_initial, jac=real_jacobian, method='lm',
                ftol=ftol, gtol=gtol, x_scale='jac', verbose=1,
                kwargs=model_parameters)
        for line in captured_stdout.getvalue().split('\n'):
            if line:
                logger.info(line)
        x_fit = result.x

    tf_fit = TransferFunction(
        np.hstack((x_fit[m:], np.zeros(p))),
        np.hstack((1, x_fit[:m])))
    zpk_fit = tf_fit.to_zpk()

    # restore fixed part
    zpk_full = lti_multiply(zpk_fit, zpk_fixed)

    return zpk_full


def least_squares_line_search(
        x_initial: ArrayLike,
        g_tol: float = 1e-06,
        max_outer: int = 100,
        **kwargs: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Least-squares minimization using line-search along Gauss-Newton gradient.

    Outer loop computes new search direction from Jacobian;
    inner loop conducts simple line-search along Gauss-Newton gradient.
    """
    logger = getLogger(__name__)
    x_initial = np.array(x_initial)
    m = kwargs.get('m')

    def _stabilize(x: ArrayLike, iteration: Tuple[int, int]) -> np.ndarray:
        x = np.array(x)
        a_check = np.hstack((1, x[:m]))
        a_temp = apolystab(a_check)
        if any(a_temp != a_check):
            x[:m] = a_temp[1:]
            logger.debug(
                'Stabilizing denominator at iteration %d.%d.', *iteration)
        return x

    f_initial = residuals(x_initial, **kwargs)
    e_initial = 0.5 * np.linalg.norm(f_initial)

    x_fits = np.zeros((max_outer + 1, len(x_initial)))
    x_fits[0] = x_initial
    e_fits = np.zeros((max_outer + 1, 1))
    e_fits[0] = e_initial

    message = ''
    outer = 0
    while not message:

        f_error = residuals(x_fits[outer], **kwargs)
        j_error = jacobian(x_fits[outer], **kwargs)
        j_error_squared = np.real(j_error.conj().T @ j_error)
        j_error_f_error = np.real(j_error.conj().T @ f_error)
        try:
            dx_gauss_newton = np.linalg.solve(
                j_error_squared, -j_error_f_error)
        except np.linalg.LinAlgError as ex:
            message = str(ex)
            break

        if (np.logical_not(np.isreal(dx_gauss_newton)).any() or
                np.isinf(dx_gauss_newton).any() or
                np.isnan(dx_gauss_newton).any()):
            message = 'Gradient contains Inf or NaN or is imaginary.'
            break

        if all(np.abs(dx_gauss_newton) <= g_tol*np.abs(x_fits[outer])):
            message = 'Minimum gradient reached.'
            break

        # line search along the Gauss-Newton gradient
        alpha = 1.0
        x_new = x_fits[outer] + alpha * dx_gauss_newton

        x_new = _stabilize(x_new, (outer, 0))
        f_new = residuals(x_new, **kwargs)
        e_new = 0.5 * np.linalg.norm(f_new)

        inner = 0
        # take the largest step that does not increase the error
        while e_new >= e_fits[outer] and not message:
            alpha /= 2
            x_new = x_fits[outer] + alpha * dx_gauss_newton

            x_new = _stabilize(x_new, (outer, inner))

            # compute transfer function and fit error
            f_new = residuals(x_new, **kwargs)
            e_new = 0.5 * np.linalg.norm(f_new)

            inner += 1
            if inner == 10:
                dx_gauss_newton = j_error_f_error / \
                    np.linalg.norm(j_error_squared) * len(j_error_squared)
                alpha = 2.0

            if inner == 20:
                message = 'Minimum coefficent increment.'

        outer += 1
        x_fits[outer] = x_new
        e_fits[outer] = e_new

        if outer == max_outer:
            message = 'Maximum number of iterations.'

    x_fits = np.array(x_fits[:outer])
    e_fits = e_fits[:outer]
    return x_fits, e_fits, message


def apolystab(
    polynomial: ArrayLike,
) -> np.ndarray:
    """Return stabilized denominator polynomial of real analog filter."""
    polynomial = np.array(polynomial)
    if polynomial.ndim != 1:
        raise ValueError('Coefficients must be a vector.')
    if polynomial.shape[0] > 0:
        roots = np.roots(polynomial)
        real_positive = np.real(roots) > 0
        if real_positive.any():
            roots[real_positive] = -roots[real_positive]
            polynomial = np.real(np.poly(roots))
    return polynomial


def zpk_out_of_band(
    system: ZerosPolesGain,
    f: ArrayLike,
    factor_lims: Tuple[float, float] = (5, 2),
    norm_freq_hz: float = 1,
    set_sensitivity: float = 1,
) -> ZerosPolesGain:
    """
    Construct out-of-band part of nominal response, with unity gain.

    Band is expanded by configurable range of factor until a result is found
    which results in a proper transfer function after fixed part is removed.

    System will have a flat passband (near-zero phase at the normalization
    frequency) when out-of-band part is removed.

    Out-of-band response is constrained to have the sensitivity given at
    the normalization frequency.
    """
    f = np.array(f)
    f.sort()
    for factor in np.logspace(log10(factor_lims[0]),
                              log10(factor_lims[1]),
                              int(6*(factor_lims[1]/factor_lims[0]))):
        w_min = 2*np.pi*f[0]/factor
        w_max = 2*np.pi*f[-1]*factor
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

        phase_norm = phase_deg(system.freqresp(2*np.pi*norm_freq_hz)[1])
        integrations_required = -int(np.round(phase_norm/90))
        if integrations_required > 0:
            p_fixed = [0]*integrations_required + p_fixed
        else:
            z_fixed = [0]*(-integrations_required) + z_fixed

        zpk_out = lti_minreal(ZerosPolesGain(
            z_fixed, p_fixed, set_sensitivity /
            sensitivity(ZerosPolesGain(z_fixed, p_fixed, 1), norm_freq_hz)))

        if lti_is_proper(lti_divide(system, zpk_out)):
            break

    return zpk_out


def get_weights(
    weighting: Sequence[str],
    f_meas: ArrayLike,
    var_meas: ArrayLike,
    h_initial: ArrayLike,
    var_lims: Tuple[float, float]
) -> np.ndarray:
    """Construct various kinds of weighting schemes."""
    var_meas = np.array(var_meas)
    unsupported = [item for item in weighting if item not in WEIGHTING_SCHEMES]
    if any(unsupported):
        raise ValueError(
            f'Weighting {unsupported} not in supported weighting schemes: '
            f'{WEIGHTING_SCHEMES}')

    weights = np.ones_like(f_meas)
    if 'variance' in weighting:
        var_meas[var_meas < var_lims[0]] = var_lims[0]
        var_meas[var_meas > var_lims[1]] = np.Inf
        weights /= np.sqrt(var_meas)
    if 'response' in weighting:
        weights /= np.abs(h_initial)
    if 'frequency' in weighting:
        weights /= np.sqrt(f_meas)

    return weights


def _plot_possible_weights(
    weighting: Sequence[str],
    f_meas: ArrayLike,
    var_meas: ArrayLike,
    h_initial: ArrayLike,
    var_lims: Tuple[float, float]
) -> None:
    """Plot all possible weighting schemes. Same args as get_weights()."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    example_weightings = []
    for num in range(len(WEIGHTING_SCHEMES) + 1):
        example_weightings += list(combinations(WEIGHTING_SCHEMES, num))
    for ex_weighting in example_weightings:
        ex_weights = get_weights(
            ex_weighting, f_meas, var_meas, h_initial, var_lims)
        width = 3 if set(ex_weighting) == set(weighting) else 1.5
        ax.loglog(f_meas, ex_weights, label=','.join(ex_weighting) or 'none',
                  linewidth=width)
    ax.set_ylabel('Weight')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Frequency [Hz]')
    weighting_png = Path(__file__).stem + '_weighting.png'
    getLogger(__name__).info('Writing: %s', weighting_png)
    fig.savefig(weighting_png, bbox_inches='tight')
