# -*- coding: utf-8 -*-
'General-Purpose utilities.'
# pylint: disable=logging-not-lazy
# for Python 2 & 3 compatibility
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from math import floor, log10
from operator import mul
from functools import reduce, wraps
import argparse
from pdb import post_mortem
from traceback import print_exception

import numpy as np

# for Python 2 & 3 compatible unicode support
from past.builtins import basestring


# %% argument parsing
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    """Preserve linefeeds in docstring and include default values in help."""


class MyArgumentParser(argparse.ArgumentParser):
    """Trigger printing of help on any argument parsing error."""

    def error(self, message):
        """Display help and exit."""
        self.print_help()
        sys.exit('Error parsing arguments: ' + message + '\n')


def string_list(argument):
    """Turn an argument which might be a string or a tuple into a list."""
    if isinstance(argument, basestring) or argument is None:
        argument = [argument]
    elif isinstance(argument, tuple):
        argument = list(argument)
    return argument


# %% formatting
def round_sig(value, num_significant=3):
    """
    Round to a given number of significant figures.

    Not the same as rounding to a number of digits.
    """
    if value is None or value == 0 or np.isinf(value) or np.isnan(value):
        return value

    order_of_magnitude = int(floor(log10(abs(value))))
    num_digits = num_significant - order_of_magnitude - 1

    return np.round(value, num_digits)


# pylint: disable=too-many-arguments
def pretty_units(input_value, input_unit, units, factors, fmt, thresh=0.95):
    """
    Convert a value with specified units to string in related units.

    The argument fmt can be a string format specifier,
    or an integer number of significant digits.
    """
    input_unit_index = next((i for i, unit in enumerate(units)
                             if input_unit == unit), None)
    if input_unit_index is None:
        raise ValueError('Input unit "%s" not supported' % input_unit)

    sign = np.sign(input_value)
    input_value = abs(input_value)
    base_multiplier = reduce(mul, factors[:input_unit_index], 1.)
    multipliers = [base_multiplier/reduce(mul, [factor
                                                for factor in factors[:n]], 1.)
                   for n in range(len(units))]

    for unit, multiplier in zip(units, multipliers):
        value = input_value/multiplier
        if value > thresh:
            if isinstance(fmt, int):
                value = round_sig(value, fmt)
                fmt = '%g'
            return fmt % (sign*value) + unit

    return '%g' % value + input_unit


def parse_units(string, output_unit, units, factors):
    """Convert a string including units to a numeric value in related units."""
    output_unit_index = next((i for i, unit in enumerate(units)
                              if output_unit == unit), None)
    if output_unit_index is None:
        raise ValueError('Output unit "%s" not supported' % output_unit)

    input_unit = sorted([unit for unit in units if unit in string],
                        key=len)[-1]
    try:
        input_value = float(string.replace(input_unit, ''))
    except ValueError:
        return None

    input_unit_index = next((i for i, unit in enumerate(units)
                             if input_unit == unit), None)

    multipliers = [reduce(mul, [factor for factor in factors[:n]], 1.)
                   for n in range(len(units))]

    return (input_value *
            multipliers[output_unit_index] /
            multipliers[input_unit_index])


TIME_UNITS = ['y', 'w', 'd', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps']
TIME_FACTORS = [52, 365.25/52, 24, 60, 60, 1e3, 1e3, 1e3]


def pretty_duration(value, input_unit='s', fmt=3, thresh=0.95):
    """Convert a duration to a string which includes units."""
    return pretty_units(value, input_unit, TIME_UNITS, TIME_FACTORS, fmt,
                        thresh)


def parse_duration(string, output_unit='s'):
    """Convert a string to a duration in specified units."""
    return parse_units(string, output_unit, TIME_UNITS, TIME_FACTORS)


BINARY_BYTE_UNITS = ['TiB', 'GiB', 'MiB', 'KiB', 'B']
BINARY_BYTE_FACTORS = [2**10]*(len(BINARY_BYTE_UNITS) - 1)

BYTE_UNITS = ['TB', 'GB', 'MB', 'KB', 'B']
BYTE_FACTORS = [1e3]*(len(BYTE_UNITS) - 1)


def pretty_bytes(value, input_unit='B', fmt=3, style='binary'):
    """Convert a file size to a string which includes units."""
    if style == 'binary':
        return pretty_units(value, input_unit, BINARY_BYTE_UNITS,
                            BINARY_BYTE_FACTORS, fmt)

    return pretty_units(value, input_unit, BYTE_UNITS, BYTE_FACTORS, fmt)


VOLTAGE_UNITS = ['MV', 'kV', 'V', 'mV', 'uV', 'nV']
VOLTAGE_FACTORS = [1e3]*(len(VOLTAGE_UNITS) - 1)


def parse_bytes(string, output_unit='B'):
    """Convert a string to a size in specified units."""
    result = parse_units(string, output_unit, BINARY_BYTE_UNITS,
                         BINARY_BYTE_FACTORS)
    if result is None:
        result = parse_units(string, output_unit, BYTE_UNITS, BYTE_FACTORS)

    return result


def pretty_voltage(value, input_unit='V', fmt=3):
    """Convert a voltage to a string which includes units."""
    return pretty_units(value, input_unit, VOLTAGE_UNITS, VOLTAGE_FACTORS, fmt)


def parse_voltage(string, output_unit='V'):
    """Convert a string to a voltage in specified units."""
    return parse_units(string, output_unit, VOLTAGE_UNITS, VOLTAGE_FACTORS)


def to_string_no_index(df, **kwargs):
    """
    Work around issue with pandas.DataFrame.to_string().

    This is a workaround for a bug in pandas 0.18 whereby if you specify
    index=False then column justification is lost.

    See https://github.com/pydata/pandas/issues/13032
    """
    index_width = max([len(str(i)) for i in df.index]) + 1
    lines = df.to_string(**kwargs).split('\n')
    string = '\n'.join([line[index_width:] for line in lines]) + '\n'
    return string


# %% logarithmic binning
def preferred_number(value, series=(1, 2, 5), method='nearest'):
    """Find nearest value from Renard or E-series."""
    # ensure preferred value series ends a factor of 10 higher than it starts
    if series[-1] != 10*series[0]:
        series = np.hstack((series, 10*series[0]))

    mul_series = 10**(np.floor(np.log10(series[0])))
    log_series = np.log10(series) % 1.
    log_series[-1] += 1

    sign_value = np.sign(value)
    value = np.abs(value)
    mul_value = 10**(np.floor(np.log10(value)))
    log_value = np.log10(value) % 1.

    diffs = log_series - log_value
    if method == 'floor':
        diffs[diffs > 0] = 1
    elif method == 'ceil':
        diffs[diffs < 0] = 1

    nearest = sign_value*series[np.argmin(np.abs(diffs))]*mul_value/mul_series
    return nearest


R_SERIES = {
    3: [1, 2, 5, 10],
    5: [1.00, 1.60, 2.50, 4.00, 6.30, 10.00],
    10: [1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.00],
    20: [1.00, 1.12, 1.25, 1.40, 1.60, 1.80, 2.00, 2.24, 2.50, 2.80,
         3.15, 3.55, 4.00, 4.50, 5.00, 5.60, 6.30, 7.10, 8.00, 9.00, 10.00],
    40: [1.00, 1.06, 1.12, 1.18, 1.25, 1.32, 1.40, 1.50, 1.60, 1.70,
         1.80, 1.90, 2.00, 2.12, 2.24, 2.36, 2.50, 2.65, 2.80, 3.00,
         3.15, 3.35, 3.55, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.30,
         5.60, 6.00, 6.30, 6.70, 7.10, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00],
    80: [1.00, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.22, 1.25, 1.28,
         1.32, 1.36, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75,
         1.80, 1.85, 1.90, 1.95, 2.00, 2.06, 2.12, 2.18, 2.24, 2.30,
         2.36, 2.43, 2.50, 2.58, 2.65, 2.72, 2.80, 2.90, 3.00, 3.07,
         3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.87, 4.00, 4.12,
         4.25, 4.37, 4.50, 4.62, 4.75, 4.87, 5.00, 5.15, 5.30, 5.45,
         5.60, 5.80, 6.00, 6.15, 6.30, 6.50, 6.70, 6.90, 7.10, 7.30,
         7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00],
}


E_SERIES = {
    6: [10, 15, 22, 33, 47, 68, 100],
    12: [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100],
    24: [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
         33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91, 100],
    48: [100, 105, 110, 115, 121, 127, 133, 140, 147, 154, 162, 169,
         178, 187, 196, 205, 215, 226, 237, 249, 261, 274, 287, 301,
         316, 332, 348, 365, 383, 402, 422, 442, 464, 487, 511, 536,
         562, 590, 619, 649, 681, 715, 750, 787, 825, 866, 909, 953, 1000],
    96: [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130,
         133, 137, 140, 143, 147, 150, 154, 158, 162, 165, 169, 174,
         178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232,
         237, 243, 249, 255, 261, 267, 274, 280, 287, 294, 301, 309,
         316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
         422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549,
         562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
         750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976, 1000],
    192: [100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 113, 114,
          115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132,
          133, 135, 137, 138, 140, 142, 143, 145, 147, 149, 150, 152,
          154, 156, 158, 160, 162, 164, 165, 167, 169, 172, 174, 176,
          178, 180, 182, 184, 187, 189, 191, 193, 196, 198, 200, 203,
          205, 208, 210, 213, 215, 218, 221, 223, 226, 229, 232, 234,
          237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 271,
          274, 277, 280, 284, 287, 291, 294, 298, 301, 305, 309, 312,
          316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 357, 361,
          365, 370, 374, 379, 383, 388, 392, 397, 402, 407, 412, 417,
          422, 427, 432, 437, 442, 448, 453, 459, 464, 470, 475, 481,
          487, 493, 499, 505, 511, 517, 523, 530, 536, 542, 549, 556,
          562, 569, 576, 583, 590, 597, 604, 612, 619, 626, 634, 642,
          649, 657, 665, 673, 681, 690, 698, 706, 715, 723, 732, 741,
          750, 759, 768, 777, 787, 796, 806, 816, 825, 835, 845, 856,
          866, 876, 887, 898, 909, 920, 931, 942, 953, 965, 976, 988, 1000]
}


def stdval(value, num=96, bump=0, preferred=None):
    '''
    Compute nearest values in a standard-value series.

    For non-standard E-numbers, Renard numbers are used. Note that the
    standard E-series do not strictly follow the Renard number series,
    which is why lookup tables must be used. An optional variable "bump"
    specifies the number by which the series index is to be adjusted up or
    down, and is useful for ceiling/floor type operations.

    Arguments
    ---------
    :param value: Values to which the nearest standard values are sought
    :param num: Number of preferred values in standard series,
        e.g. 96 for E96 series
    :type num: list[int]
    :param preferred: Preferred values, e.g. [1, 2, 5, 10]
    :type preferred::mod:numpy:array:
    :param float bump: Amount by which each series index is adjusted before
        choosing value

    Note that num=3 gives same result as preferred=[1, 2, 5, 10], but
    num=24 would not give the correct E24 series if it weren't overridden.

    :returns output: Nearest standard values after rounding
    '''
    # we're going to need to do some elementwise operations
    x_type = type(value)
    value = np.asarray(value, dtype=float)

    # and some operations which depend on input being a column vector
    x_shape = value.shape
    value = np.reshape(value, (value.size, 1))

    # negative values will not be handled
    value[value < 0] = np.nan

    if preferred is None:
        if num in E_SERIES.keys():
            preferred = np.array(E_SERIES[num])
        elif num in R_SERIES.keys():
            preferred = np.array(R_SERIES[num])
        log_series = True
    else:
        # for the purpose of "bumping" it will be assumed that the preferred
        # values are approximately logarithmically-spaced and span a decade
        preferred = np.asarray(preferred, dtype=float)
        preferred = np.reshape(preferred, (1, preferred.size))
        num = preferred.size - 1
        log_series = False

    # bump input up or down as requested to support rounding up and down
    if log_series:
        value = value*10**(np.asarray(bump)/num)

    if preferred is not None:
        # determine how many digits of result to keep
        preferred = np.asarray(preferred, dtype=float)
        preferred = np.reshape(preferred, (1, preferred.size))
        digits = len('%d' % preferred[0][0])

        # compute multiplier for rounding
        multiplier = 10**np.floor(np.log10(value) - digits + 1)

        # shift input to have the right number of digits
        value = value/multiplier

        # find nearest standard value in a logarithmic sense
        pref_mat = np.tile(np.log10(preferred), (value.size, 1)).transpose()
        x_mat = np.tile(np.log10(value), (1, preferred.size)).transpose()

        log_dist = pref_mat - x_mat
        i_closest = np.argmin(np.abs(log_dist), 0)

        if log_series or bump == 0:
            output = preferred[0, i_closest]
        else:
            # for non-logarithmic series, bump just means round up or down
            if bump > 0 and log_dist[i_closest] < 0:
                if i_closest == preferred.size:
                    output = preferred[1]*10
                else:
                    output = preferred[i_closest + 1]

            elif bump < 0 and log_dist[i_closest] > 0:
                if i_closest == 1:
                    output = preferred[-1]/10
                else:
                    output = preferred[i_closest - 1]

        # restore correct number
        output = output[:, None]*multiplier

    else:
        # the nth power of a decade is the base
        base = 10**(1.0/num)

        # determine how many digits of result to keep
        digits = np.max((1, -np.round(np.log10(base - 1) - 1.5)))

        # compute the sequence number
        exponent = np.round(np.log(value)/np.log(base))

        # compute raw result
        raw = base**exponent

        # compute multiplier for rounding
        multiplier = 10**np.floor(np.log10(raw) - digits + 1)

        # round result to requested number of digits
        output = np.round(raw/multiplier)*multiplier

    output = np.reshape(output, x_shape)

    if (x_type is int) | (x_type is float):
        output = float(output)

    return output


def logspace(start, stop, num=12):
    """
    Compute logarithmically spaced vector of preferred numbers.

    See stdval.
    """
    log_start = np.floor(np.log10(start))
    log_stop = np.ceil(np.log10(stop))
    n_total = int(num*(log_stop-log_start)) + 1
    temp = stdval(np.logspace(log_start, log_stop, num=n_total), num=num)
    return temp[np.bitwise_and(temp >= start, temp <= stop)]


# %% debugging
def debug_on(*exceptions):
    """Decorate unittest function so that debugger is invoked on exceptions."""
    if not exceptions:
        exceptions = (AssertionError, )

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                print_exception(*info)
                post_mortem(info[2])
        return wrapper

    return decorator
