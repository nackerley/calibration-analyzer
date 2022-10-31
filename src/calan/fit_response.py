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

TODO: give complete formula for Jacobian

@nackerle
"""
from typing import Optional, Tuple
from logging import getLogger

import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike
from scipy.signal import ZerosPolesGain, TransferFunction
from scipy.optimize import least_squares, OptimizeResult

# from control import minreal


def fit_response(
    zpk_nom: ZerosPolesGain,
    f_meas: ArrayLike,
    h_meas: ArrayLike,
    var_meas: Optional[ArrayLike] = None,
    zpk_fixed: Optional[ZerosPolesGain] = None,
    gtol: float = 1e-06,
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

    Outputs:
      - zpk_fit:   best_fit transfer function
      - result:    optimization result

    TODO Determine whether denominator must be constrained to be stable.

    TODO support different weighting schemes:
      - 'variance" 1/sqrt(var) to account for measurement errors
      - 'response' 1/abs(h_nom) to emphasize importance of passband
      - 'frequency' 1/f to account for over-weighting of high frequencies by FFT
    """
    logger = getLogger(__name__)
    if var_meas is None:
        var_meas = np.ones_like(f_meas)

    f_meas = np.array(f_meas)
    h_meas = np.array(h_meas)
    var_meas = np.array(var_meas)
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

    if zpk_fixed is not None:
        logger.info(
            'Removing fixed part from measured response and initial guess')
        # zpk_nom = minreal(zpk_nom / zpk_fixed)
        zpk_nom = zpk_nom / zpk_fixed
        h_meas /= zpk_fixed.freqresp(2*np.pi*f_meas)

    # remove cancelling poles and zeros from initial guess
    zpk_nom = zp_cancel(zpk_nom)
    tf_nom = zpk_nom.to_tf()

    # extract coefficient vector
    a_guess = tf_nom.den
    b_guess = tf_nom.num
    if a_guess[0] == 0:
        logger.warning('Trimming leading zeroes from denominator coefficients.')
        a_guess = np.trim_zeros(a_guess, trim='f')
    if b_guess[0] == 0:
        logger.warning('Trimming leading zeroes from numerator coefficients.')
        b_guess = np.trim_zeros(b_guess, trim='f')

    last_non_zero = np.nonzero(b_guess)[1][-1]
    integrator_count = b_guess.shape[0] - last_non_zero
    logger.info('Fixing %d zeros at zero.', integrator_count)
    b_guess = b_guess[:last_non_zero]

    if a_guess[0] != 1:
        logger.warning('Normalizing denominator of initial guess.')
        b_guess = b_guess / a_guess[0]
        a_guess = a_guess / a_guess[0]

    a_temp = apolystab(a_guess)
    if any(a_temp != a_guess):
        logger.warning('Stabilizing denominator of initial guess.')
        a_guess = np.copy(a_temp)

    # determine fitting orders
    num_order = b_guess.shape[0] + integrator_count
    den_order = a_guess.shape[0]
    measuremen_count = f_meas.shape[0]

    # ensure system is overdetermined
    if den_order + num_order - integrator_count > (var_meas < 10).sum():
        raise ValueError('There must be more measurements than coefficients.')

    # initial guess
    x_initial = np.hstack((a_guess[1:], b_guess))

    # precompute all required powers of frequencies
    max_power = max(den_order, num_order)
    omega = np.ones(measuremen_count, max_power + 1)
    w_meas = 2*np.pi*f_meas
    for i in range(max_power):
        omega[:, max_power - i] = 1j*w_meas*omega[:, max_power - i + 1]

    # subsets needed for computation of model and Jacobian
    omega_m = omega[:, max_power - den_order]
    omega_a = omega[:, max_power - den_order + 1:max_power]
    omega_b = omega[:, max_power - num_order:max_power - integrator_count]

    def model(x):
        """Compute fitted transfer function model as function of frequency."""
        return omega_b @ x[den_order:] / (omega_m + omega_a @ x[:den_order])

    # variance- and response-based weights
    wt_meas = 1.0 / np.sqrt(var_meas) / np.abs(model(x_initial))

    def residuals(x):
        """Compute weighted mean squared error over all frequencies."""
        return wt_meas * (model(x) - h_meas)

    def jacobian(x):
        """Compute partial derivatives of residuals wrt coefficients."""
        # TODO: check if broadcasting works as well as explicit tiling
        part_a = -omega_a * np.tile(
            model(x) / (omega_m + omega_a @ x[:den_order]), (1, den_order))
        part_b = omega_b / np.tile(
            omega_m + omega_a @ x[:den_order], (1, num_order + 1 - integrator_count))
        return np.hstack((part_a, part_b)) * np.tile(
            wt_meas, (1, den_order + num_order + 1 - integrator_count))

    result = least_squares(
        residuals, x_initial, jac=jacobian, method='lm', gtol=gtol,
        x_scale='jac', verbose=1)

    x_fit = result.x
    b_fit = np.hstack((x_fit[:num_order], np.zeros(integrator_count)))
    a_fit = np.hstack((1, x_fit[num_order:]))
    tf_fit = TransferFunction(a_fit, b_fit)
    zpk_fit = tf_fit.to_zpk()

    return zpk_fit, result


def old_least_squares(x_fit, e_fit, model, residuals, error, gradient,
                      max_incr, max_iter, den_order):
    """Old hand-crafted minimization loop."""
    logger = getLogger(__name__)
    report = True
    iteration = -1
    stop = 0
    while not stop:
        iteration = iteration + 1
        dx_gauss_newton, j_error_f_error, j_error_squared = gradient(
            x_fit[iteration - 1])

        if (np.logical_not(np.isreal(dx_gauss_newton)).any() or
                np.isinf(dx_gauss_newton).any() or
                np.isnan(dx_gauss_newton).any()):
            stop = 4
            iteration = iteration - 1
            break

        if all(np.abs(dx_gauss_newton) <= np.abs(x_fit[:, iteration])/max_incr + 1):
            stop = 1

        def stabilize_and_report_first(x, report):
            a_check = np.hstack((1, x[:den_order, 0].T))
            a_temp = apolystab(a_check)
            if any(a_temp != a_check):
                x[:den_order, 0] = a_temp[1:].T  # FIXME: unnecessary transpose?
                if report:
                    logger.info('Stabilizing denominator in loop.')
                    report = False
            return report

        # line search along the Gauss-Newton gradient
        alpha = 1.0
        x_new = x_fit[:, iteration] + alpha * dx_gauss_newton

        report = stabilize_and_report_first(x_new, report)

        h_new = model(x_new)
        f_new = residuals(h_new)
        e_new = error(f_new)

        i_search = 0
        while e_new >= e_fit[iteration] and not stop:
            i_search = i_search + 1
            alpha = alpha / 2
            x_new = x_fit[:, iteration] + alpha * dx_gauss_newton

            report = stabilize_and_report_first(x_new, report)

            # compute transfer function and fit error
            h_new = model(x_new)
            f_current = residuals(h_new)
            e_new = error(f_current)

            if i_search == 10:
                dx_gauss_newton = j_error_f_error / norm(j_error_squared) * len(j_error_squared)
                alpha = 2.0

            # after twenty steps quit
            if i_search == 20:
                x_new = x_fit[:, iteration]
                stop = 2

        x_fit[:, iteration + 1] = x_new
        e_fit[iteration + 1] = e_new

        logger.debug(
            '%d line search iterations on step %d.', i_search, iteration)
        if iteration == max_iter:
            stop = 3

    return iteration, stop, dx_gauss_newton, j_error_f_error, j_error_squared


def apolystab(
    polynomial: np.ndarray,
) -> np.ndarray:
    """Return stabilized denominator polynomial of real analog filter."""
    if polynomial.shape[0] != 1:
        raise ValueError('Denominator coefficients must be row vector.')
    if polynomial.shape[1] > 0:
        roots = np.roots(polynomial)
        real_positive = np.real(roots) > 0
        if real_positive.any():
            roots[real_positive] = -roots[real_positive]
            polynomial = np.real(np.poly(roots))
    return polynomial


def zp_cancel(
    old: ZerosPolesGain,
    tolerance: float = 0.001,
    f_norm: float = 1,
) -> ZerosPolesGain:
    """
    Remove nearly-equal zero-pole pairs from a transfer function.

    Pole-zero pairs must be farther apart in order to be considered
    cancelling when the damping is low.  A pole-zero pair can therefore
    cancel if the following condition is met:

    np.abs(z-p) / np.sqrt(np.abs(z)*np.abs(p))
        / np.sqrt(np.abs(np.cos(np.angle(z))*np.cos(np.angle(p))))
        < tolerance
    """
    if not old.zeros or not old.poles:
        return old.copy()

    zeros = np.tile(old.zeros.T, old.poles.shape)
    poles = np.tile(old.poles, old.zeros.T.shape)

    if tolerance == 0:
        cancels = poles == zeros

        # retain only first cancellation
        for i in np.arange(cancels.shape[0]).reshape(-1):
            j = np.argmax(cancels[i, :], axis=1)
            if j:
                cancels[i, j + 1:] = False
                cancels[i + 1:, j] = False
    else:
        cancelling = (
            np.abs(zeros - poles) / np.sqrt(np.abs(poles)*np.abs(zeros)) /
            np.sqrt(np.abs(np.cos(np.angle(poles))) *
                    np.abs(np.cos(np.angle(zeros)))))
        cancelling[np.isnan(cancelling)] = 0

        # retain only closest cancellation
        cancels = np.full_like(cancelling, False)

        while (cancelling < tolerance).any():
            i, j = np.unravel_index(np.argmin(cancelling), cancelling.shape)
            cancelling[i, :] = np.Inf
            cancelling[:, j] = np.Inf
            cancels[i, j] = True
            cancelling = np.ma.array(cancelling, cancels)

    z_new = old.zeros(~cancels.any(axis=0))
    p_new = old.poles(~cancels.any(axis=1).T)
    new = ZerosPolesGain(z_new, p_new, 1)
    k_new = (old.freqresp(2*np.pi*f_norm).abs() /
             new.freqresp(2*np.pi*f_norm).abs())  # pylint: disable=no-member
    new = ZerosPolesGain(z_new, p_new, k_new)

    return new
