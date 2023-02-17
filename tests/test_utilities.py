"""
Test some general-purpose utilities.
"""
# pylint: disable=missing-function-docstring
from numpy import pi

from calan.utilities import (
    pretty_duration, parse_duration,
    pretty_bytes, parse_bytes,
    pretty_voltage, parse_voltage,
    round_sig)


def test_pretty_duration():
    assert pretty_duration(3600, 's') == '1h'
    assert pretty_duration(100e6, 'us') == '1.67m'


def test_parse_duration():
    assert parse_duration('1h', 's') == 3600
    assert parse_duration('1.67m', 'us') == 1.67*60e6


def test_pretty_bytes():
    assert pretty_bytes(123456789) == '118MiB'
    assert pretty_bytes(1234567, style='decimal') == '1.23MB'


def test_parse_bytes():
    assert parse_bytes('118MiB') == 123731968
    assert parse_bytes('1.23MB') == 1230000


def test_pretty_voltage():
    assert pretty_voltage(pi) == '3.14V'
    assert pretty_voltage(60e12, 'uV') == '60MV'


def test_parse_voltage():
    assert parse_voltage('3.14V', 'V') == 3.14
    assert parse_voltage('60MV', 'uV') == 60e12


def test_round_sig():
    assert round_sig(pi, 1) == 3
    assert round_sig(0.001*pi, 2) == 0.0031
    assert round_sig(0.1*pi) == 0.314
    assert round_sig(100*pi, 4) == 314.2
