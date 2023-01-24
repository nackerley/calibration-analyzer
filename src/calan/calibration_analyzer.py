#!/usr/bin/env python3
"""
Analyzer for broadband calibrations of seismometers using arbitrary signals.

Calibration output waveforms, station metadata and calibration input response
metadata must be provided.

Notes
-----
1. Transfer function model terminology:
  * `sensor`: from ground motion (m/s or m/s^2) to V
  * `cal`: from V to ground motion, typically m/s^2
  * `system` product of `sensor` and `cal`, dimensionless and typically
    having a slope proportional to 1/f in the passband.

2. The length of FFT segments can be specified directly, and is efficient
for any number with a large number of small factors (not just powers
of 2), but it is generally more convenient to specify a target number
of segments as it is the (actual) number of segments in Welch's method
which reduces the variance in the result.

3. Not all of the calibration input response metadata is used. In particular,
network, station, location and channel codes are ignored. Only the first two
stages of the response are retained. The first is assumed to be the sensor
poles and zeros, while the second gives the digitial-to-analog conversion
sensitivity.

Authors
-------
nicholas.ackerley@nrcan-rncan.gc.ca
"""
# pylint: disable=too-many-lines
from __future__ import annotations

import os
import sys
import lzma
import logging.config
import warnings
from io import StringIO
from contextlib import redirect_stdout
from glob import glob
from operator import mul
from functools import reduce
from math import log10, floor, ceil
from tempfile import gettempdir
from typing import Optional, Sequence, Tuple

from scipy import special
from scipy.signal import lti, BadCoefficients, ZerosPolesGain
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import pandas as pd
import statsmodels.api as sm

from obspy import read, read_inventory, Trace, Stream, UTCDateTime
from obspy.core.inventory import Inventory, Response, ResponseStage
from obspy.signal.invsim import simulate_seismometer

from calan.core import (
    PACKAGE, VERSION, factor_names, subplots_squeeze, zpk_cancel,
    zpk_from_zpsf, unwrap_mid, extract_decimation_coefficients, multi_decim,
    recompute_normalization_factors)
from calan.utilities import (
    logspace, pretty_duration, round_sig, str_sig, MyArgumentParser, MyFormatter)
from calan.stft import Stft, len_fft_welch, num_windows_welch
from calan.calibration_toolbox import (
    sample_hold_digitize, pad_for_decimation)
from calan.fit_response import fit_response, zpk_divide, zpk_out_of_band

# setup
warnings.simplefilter('error', category=BadCoefficients)
pd.plotting.register_matplotlib_converters()
np.set_printoptions(suppress=True, precision=6)

# defaults

# analysis setup
DEFAULT_NUM_WINDOWS = 30
DEFAULT_WINDOW = 'hann'

# inputs
DEFAULT_OUTPUT_PATTERN = '*.mseed'
DEFAULT_RESPONSE_PATTERN = '*.resp'
DEFAULT_CAL_SIGNAL_FILE = 'prb_1V_10ms_3h.lzma'
DEFAULT_CAL_RESPONSE_FILE = 'Centaur_Trillium120Q_Calibration.xml'
DEFAULT_DPI = 120
DEFAULT_ORIENTATION_MAP = ('', '')

# Centaur configuration
DEFAULT_LEAD_IN = 360
DEFAULT_LEAD_OUT = 360
DEFAULT_PRE_TIME = 10
DEFAULT_POST_TIME = 10
DEFAULT_DELAY_START = 0
DEFAULT_DISCARD = 0

# constants
THIS_FILE_NAME = os.path.basename(__file__)
LOG_FILE_NAME = os.path.splitext(THIS_FILE_NAME)[0] + '.log'

# Guralp digitizers produce separate miniseed for calibration input and output
OUTPUT_FLAG = '_Output'
INPUT_FLAG = '_Input'
CAL_COMPONENT = 'C'

# Nanometrics Centaur User Guide 17935R5, 2016-11-02
DAC_BITS = 16
CALIBRATION_DTYPE = f'int{DAC_BITS}'

# IDC-ENG-SPC-103-Rev.7.3, May 2017
TEST_BAND_HZ = (0.02, 16)
MAX_AMPLITUDE_PERCENT = 5
MAX_PHASE_DEGREES = 5

CAL_RESULT_MSG_ID = 'CAL_{year:4d}_1_{station}_cr'
CAL_RESULT_REF_ID = 'CAL_{year:4d}_1_{station}_cs'
IMS_DATETIME_FMT = '%Y/%m/%d %H:%M'

IMS_HEADER = '''\
begin IMS2.0
msg_type COMMAND_RESPONSE
msg_id {msg_id}
ref_id {ref_id}
time_stamp {time_stamp}
'''
RESPONSE_HEADER = '''
sta_list {station}
chan_list {channel}
CALIBRATE_RESULT
CALIB {calib:.6g}
CALPER {calper}
in_spec {in_spec}
data_type RESPONSE IMS2.0
'''
CAL_BLOCK = (
    'CAL2 {station:5.5s} {channel:3.3s} {aux_id:4.4s} {inst_type:6.6s} '
    '{calib:15.8e} {calper:7.3f} {sample_rate:11.5f} {start} {end}' + '\n')
FAP_HEADER = (
    'FAP2 {stage:2d} {units:1.1s} {decimation:4.4s} {group_correction:8.3f} '
    '{count:3d} {description:25.25s}' + '\n')
FAP_DATA = ' {frequency:10.5f} {amplitude:15.8e} {phase:4.0f}' + '\n'
MAX_FAP_LEN = 999
IMS_FOOTER = 'stop' + '\n'

OUTPUT_UNIT_MAP = {
    "DISP": ["M"],
    "VEL": ["M/S", "M/SEC"],
    "ACC": ["M/S**2", "M/(S**2)", "M/SEC**2", "M/(SEC**2)",
            "M/S/S"]}
MOTION = {}
UNITS = {}
for motion, motion_units in OUTPUT_UNIT_MAP.items():
    UNITS[motion] = motion_units[0]
    for unit in motion_units:
        MOTION[unit] = motion
ORDER = {'ACC': 0, 'VEL': 1, 'DISP': 2}

CHECK_PERCENT = 0.01
CHECK_CLIP = 8000000
PLOT_CHOICES = ['basic', 'diagnostic']


# definitions
def _argparser() -> MyArgumentParser:
    """Command-line interface."""
    parser = MyArgumentParser(prog=os.path.splitext(THIS_FILE_NAME)[0],
                              description=__doc__,
                              formatter_class=MyFormatter)

    parser.add_argument(
        '-g', '--pattern', default=DEFAULT_OUTPUT_PATTERN,
        help='glob pattern matching calibration output files, '
        'in any format readable by obspy.read()')
    parser.add_argument(
        '-n', '--num-windows', default=DEFAULT_NUM_WINDOWS, type=int,
        help="number of windows to used for Welch's method")
    parser.add_argument(
        '--len-fft', default=None, type=int,
        help="window length for Welch's method, overrides num_windows")
    parser.add_argument(
        '-w', '--window', default=DEFAULT_WINDOW,
        help="window function to be used for Welch's method")
    parser.add_argument(
        '-r', '--response-pattern', default=DEFAULT_RESPONSE_PATTERN,
        help='station metadata in any format readable by '
        'obspy.read_inventory()')
    parser.add_argument(
        '-c', '--calibration-signal-file', default=DEFAULT_CAL_SIGNAL_FILE,
        help='calibration input file, lzma compressed')
    parser.add_argument(
        '-i', '--calibration-response-file',
        default=DEFAULT_CAL_RESPONSE_FILE,
        help='calibration input response in any format readable by '
        'obspy.read_inventory().')
    parser.add_argument(
        '-d', '--delay-start', default=DEFAULT_DELAY_START, type=float,
        help='amount to delay calibration start time, in seconds')
    parser.add_argument(
        '--discard-s', default=DEFAULT_DISCARD, type=float,
        help='duration to discard from start and end, in seconds')
    parser.add_argument(
        '-b', '--test-band-hz', nargs=2, type=float, default=TEST_BAND_HZ,
        metavar=('MIN_FREQUENCY_HZ', 'MAX_FREQUENCY_HZ'),
        help='frequency band, in Hz, over which to apply test limits')
    parser.add_argument(
        '-t', '--test-limits', nargs=2, type=float,
        default=(MAX_AMPLITUDE_PERCENT, MAX_PHASE_DEGREES),
        metavar=('MAX_AMPLITUDE_PERCENT', 'MAX_PHASE_DEGREES'),
        help='maximum deviation from nominal of amplitude in percent and '
        'phase in degrees')
    parser.add_argument(
        '-m', '--orientation-map', nargs=2, default=DEFAULT_ORIENTATION_MAP,
        help='Map external channel orientation to internal axis orientations')
    parser.add_argument(
        '--ims-instrument-type', default='',
        help='if provided, write IMS2.0 CALIBRATE_RESULT message with this type')
    parser.add_argument(
        '--fit', action='store_true',
        help='fit transfer function poles and zeros')
    parser.add_argument(
        '-p', '--plot', default='none', choices=PLOT_CHOICES,
        help='generate basic (start & end check, transfer function and '
        'variance) or diagnostic plots for each calibration')
    parser.add_argument(
        '--dpi', default=DEFAULT_DPI, type=int,
        help='resolution to use for plots in dots per inch')
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'{PACKAGE} {VERSION}')
    return parser


def feature_stats(values: ArrayLike, label: str) -> pd.DataFrame:
    """Compile statistics about real and complex poles and zeros."""
    stats = {}
    values = np.array(values).reshape(-1)
    for i, value in enumerate(values):
        if np.isreal(value):
            key = label + '_' + str(i)
            stats[key] = {
                'real_rad': np.real(value),
                'freq_hz': np.real(value)/(2*np.pi),
            }
        elif i > 0 and np.isclose(value, np.conj(values[i - 1])):
            continue
        else:
            key = label + '_' + str(i) + str(i+1)
            stats[key] = {
                'real_rad': np.real(value),
                'imag_rad': np.imag(value),
                'freq_hz': np.abs(value)/(2*np.pi),
                'damping': np.abs(np.cos(np.angle(value))),
            }

    return pd.DataFrame(stats).unstack().to_frame().T


def feature_str(values: ArrayLike, sig_dig: int = 3) -> str:
    """
    Pretty-print complex array of pole/zero features.

    Zero-values are skipped, frequencies are converted from angular to linear,
    and damping is given for complex pairs.
    """
    values = np.array(values).reshape(-1)
    non_zero = [value for value in values if value != 0]
    nonzero_unpaired = [
        value for i, value in enumerate(non_zero)
        if i == 0 or not np.isclose(value, np.conj(values[i - 1]))]

    strings = []
    for value in nonzero_unpaired:
        frequency = str_sig(np.abs(value)/(2*np.pi), sig_dig)
        string = f'{frequency} Hz'
        if np.iscomplex(value):
            damping = str_sig(np.abs(np.cos(np.angle(value))), sig_dig)
            string += f' (ζ = {damping})'
        strings.append(string)

    return ', '.join(strings)


def calibration_analyzer(
    pattern: str = DEFAULT_OUTPUT_PATTERN,
    num_windows: float = DEFAULT_NUM_WINDOWS,
    len_fft: Optional[int] = None,
    window: str = DEFAULT_WINDOW,
    response_pattern: str = DEFAULT_RESPONSE_PATTERN,
    calibration_signal_file: str = DEFAULT_CAL_SIGNAL_FILE,
    calibration_response_file: str = DEFAULT_CAL_RESPONSE_FILE,
    delay_start: float = DEFAULT_DELAY_START,
    discard_s: float = DEFAULT_DISCARD,
    ims_instrument_type: str = '',
    test_band_hz: Tuple[float, float] = TEST_BAND_HZ,
    test_limits: Tuple[float, float] = (MAX_AMPLITUDE_PERCENT, MAX_PHASE_DEGREES),
    orientation_map: Tuple[str, str] = DEFAULT_ORIENTATION_MAP,
    plot: str = '',
    fit: bool = False,
    dpi: float = DEFAULT_DPI,
) -> str:
    """Do arbitrary-signal calibration analysis."""
    logger = logging.getLogger(__name__)

    pattern_slug = ''.join(char for char in os.path.splitext(pattern)[0]
                           if char.isalnum())
    output_parts = [os.path.splitext(THIS_FILE_NAME)[0]]
    if pattern_slug:
        output_parts += pattern_slug.split('_')
    summary_csv = '_'.join(output_parts) + '.csv'

    analyzer = CalibrationAnalyzer(savefig=plot != '', dpi=dpi)

    if os.path.exists(summary_csv) and os.path.isfile(summary_csv) and \
            not os.access(summary_csv, os.W_OK):
        logger.error('Will not be able to write summary to %s.', summary_csv)
        return ''

    output_files = sorted([item for item in glob(pattern)
                           if INPUT_FLAG not in item])
    if not output_files:
        logger.error('No files match pattern "%s".', pattern)
        return ''

    dfs = []
    for output_file in output_files:
        plt.close('all')

        analyzer.load_stream(output_file, delay_start=delay_start)
        analyzer.load_response(response_pattern)
        analyzer.load_calibration_signal(calibration_signal_file)
        analyzer.load_calibration_response(calibration_response_file)
        analyzer.check_stream(discard_s=discard_s)
        analyzer.setup_nominal_responses()
        analyzer.map_orientations(orientation_map)

        if plot:
            analyzer.plot_check('start')
            analyzer.plot_check('end')

        analyzer.compute(num_windows=num_windows, len_fft=len_fft or None,
                         window=window)
        if fit:
            analyzer.fit()
        analyzer.test(test_band_hz=test_band_hz,
                      max_amplitude_percent=test_limits[0],
                      max_phase_degrees=test_limits[1])
        analyzer.estimate_errors()

        if plot:
            analyzer.plot_transfer_function(remove='system', errors='estimate')
            analyzer.plot_transfer_function(remove='cal', errors='estimate')
            analyzer.plot_variance()

        if plot == 'diagnostic':
            analyzer.plot_transfer_function(remove='', errors='')
            analyzer.plot_transfer_function(remove='system', errors='estimate',
                                            scale='linear')
            analyzer.plot_transfer_function(remove='system', errors='correct',
                                            scale='log')
            analyzer.plot_signal_to_noise()
            analyzer.plot_response('sensor', f_limits=[1e-3, 1e2])
            analyzer.plot_response('system', f_limits=[1e-3, 1e2])
            analyzer.plot_response('cal', f_limits=[1e-3, 1e2])
            analyzer.plot_simulated()
            analyzer.plot_spectrogram()

        dfs.append(analyzer.summary())

        if ims_instrument_type:
            analyzer.write_calibrate_result(ims_instrument_type)

    if not dfs:
        logger.error('No valid calibration results.')
        return ''

    df = pd.concat(dfs)

    if len(dfs) > 0:  # drop redundant nominal response rows
        df = df.loc[~df.index.duplicated(keep='last')]
    df.reset_index(col_level=1, col_fill='info', inplace=True)

    logger.info('Summary: %s', summary_csv)
    # transpose rows and columns before writing to file
    df.T.to_csv(summary_csv, float_format='%.5g')

    return summary_csv


class TimingGainFit():
    """Container for transfer function fits."""

    def __init__(
        self: TimingGainFit,
        confidence: float = 0.95,
        timing: Optional[sm.WLS] = None,
        gain: Optional[sm.WLS] = None,
    ) -> None:
        """Construct object."""
        self.confidence = confidence
        self.timing = timing
        self.gain = gain

    def __str__(self: TimingGainFit) -> str:
        """Human-readable representation."""
        lines = [self.__class__.__name__ + ':']
        if self.timing is not None:
            lines.append('    ' + self.timing_summary())
        if self.gain is not None:
            lines.append('    ' + self.gain_summary())
        return '\n'.join(lines)

    def num_sigma(self: TimingGainFit) -> float:
        """Estimte number of standard deviations from confidence interval."""
        return np.sqrt(2)*special.erfinv(self.confidence)

    def timing_summary(self: TimingGainFit) -> str:
        """Summarize timing error estimate."""
        if self.timing is None:
            return 'timing fit: None'
        uncertainty = self.num_sigma()*self.timing.bse
        digits = round(log10(abs(self.timing.params)/uncertainty)) + 1
        estimate = pretty_duration(self.timing.params, fmt=digits, thresh=0.05)
        error = pretty_duration(uncertainty, fmt=1, thresh=0.05)
        conf = 100*self.confidence
        return f'timing error {estimate} ±{error} ({conf:.0f}% conf.)'

    def gain_summary(self: TimingGainFit) -> str:
        """Summarize gain error estimate."""
        if self.gain is None:
            return 'gain fit: None'
        uncertainty = self.num_sigma()*self.gain.bse[0]
        digits = round(log10(abs(self.gain.params - 1)/uncertainty)) + 1
        estimate = round_sig(100*(self.gain.params - 1), digits)[0]
        error = round_sig(100*self.num_sigma()*self.gain.bse, 1)[0]
        conf = 100*self.confidence
        return f'gain error {estimate}% ±{error}% ({conf:.0f}% conf.)'


class CalibrationInfo():
    """Container for information about a calibration."""

    def __init__(self: CalibrationInfo) -> None:
        """Constructor."""
        self.waveform_file: str = ''
        self.response_file: Sequence[str] = []
        self.calibration_signal_file: str = ''
        self.calibration_response_file: str = ''
        self.delay_start: float = 0
        self.start: UTCDateTime = UTCDateTime(0)
        self.end: UTCDateTime = UTCDateTime(0)
        self.spec_min_freq_hz: float = np.NaN
        self.spec_max_freq_hz: float = np.NaN
        self.spec_max_amp_pct: float = np.NaN
        self.spec_max_phase_deg: float = np.NaN
        self.in_spec: Sequence[bool] = ()

    def __str__(self: CalibrationInfo) -> str:
        """Human-readable representation."""
        lines = [self.__class__.__name__ + ':']
        for key, value in self.__dict__.items():
            lines.append(f'    {key}: {value}')
        return '\n'.join(lines)


class CalibrationResponses():
    """Container for nominal and fitted responses of a calibration."""

    def __init__(self: CalibrationResponses) -> None:
        """Constructor."""
        self.sensor = ZerosPolesGain([], [], 1)
        self.cal = ZerosPolesGain([], [], 1)
        self.system = ZerosPolesGain([], [], 1)
        self.fits: Sequence[ZerosPolesGain] = []

    def __str__(self: CalibrationResponses) -> str:
        """Human-readable representation."""
        lines = [self.__class__.__name__ + ':']
        for key, value in self.__dict__.items():
            lines.append(f'    {key}')
            for line in str(value).split('\n'):
                lines.append(f'        {line}')
        return '\n'.join(lines)


class CalibrationAnalyzer():
    """An ObsPy Stream-based calibration signal analyzer."""

    CACHE_FORMAT = 'MSEED'

    def __init__(
        self: CalibrationAnalyzer,
        savefig: bool = True,
        dpi: float = DEFAULT_DPI,
    ) -> None:
        """
        Initialize calibration analyzer.

        Arguments:
        - `savefig`: Whether or not to save figures to PNG
        - `dpi`: Resolution to use when rendering figures
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.info = CalibrationInfo()
        self.stream = Stream()
        self.stft = Stft()
        self.lti = CalibrationResponses()
        self.timing_gain_fit = TimingGainFit(confidence=0.95)
        self.zpk_fits = [OptimizeResult()]
        self.savefig = savefig
        self.dpi = dpi

    def __del__(self: CalibrationAnalyzer) -> None:
        """Ensure log files are not held open."""
        logging.shutdown()

    def __str__(self: CalibrationAnalyzer) -> str:
        """Human-readable representation."""
        lines = [self.__class__.__name__ + ':']
        lines += ['    ' + line for line in str(self.info).split('\n')]
        lines += ['    ' + line for line in str(self.stream).split('\n')]
        lines += ['    ' + line for line in str(self.lti).split('\n')]
        lines += ['    ' + line for line in str(self.stft).split('\n')]
        lines += ['    ' + line for line in str(self.timing_gain_fit).split('\n')]
        return '\n'.join(lines)

    def __repr__(self: CalibrationAnalyzer) -> str:
        """Unambiguous representation."""
        return self.__str__()

    def load_stream(
        self: CalibrationAnalyzer,
        waveform_file: str,
        lead_in: float = DEFAULT_LEAD_IN,
        lead_out: float = DEFAULT_LEAD_OUT,
        pre_time: float = DEFAULT_PRE_TIME,
        post_time: float = DEFAULT_POST_TIME,
        delay_start: float = DEFAULT_DELAY_START,
    ) -> None:
        """
        Load calibration waveforms.

        Two cases are supported:
            1.  The stream does not include a calibration signal:
                The calibration start and end are inferred from lead
                in/out and event pre/post times.
            2.  The stream includes a calibration signal:
                The calibration start and end are that of the stream.
        """
        self.logger.info(waveform_file)
        self.stream = read(waveform_file)
        if OUTPUT_FLAG in waveform_file:
            input_file = waveform_file.replace(OUTPUT_FLAG, INPUT_FLAG)
            if os.path.isfile(input_file):
                self.logger.info(input_file)
                self.stream += read(input_file)
            else:
                self.logger.warning(
                    'Expected input file not found: %s', input_file)
        self.info.waveform_file = waveform_file
        self.info.start = self.stream[0].stats.starttime
        self.info.end = self.stream[0].stats.endtime

        self.stream.sort()
        calibration_trace = next((trace for trace in self.stream
                                  if trace.id[-1] == CAL_COMPONENT), None)
        if calibration_trace:
            self.stream.traces = [  # move calibration trace to end
                trace for trace in self.stream
                if trace.id != calibration_trace.id] + [calibration_trace]
        else:
            self.logger.debug(
                'lead in, out [s]: %g, %g', lead_in, lead_out)
            self.logger.debug(
                'Pre, post-event [s]: %g, %g', pre_time, post_time)
            self.logger.info(
                'Delay start [s]: %g', delay_start)
            self.info.start += lead_in + pre_time + delay_start
            self.info.end -= lead_out + post_time - delay_start
            self.info.delay_start = delay_start

    def _sampling_rate(self: CalibrationAnalyzer) -> float:
        """Return stream sampling rate."""
        return self.stream[0].stats.sampling_rate

    def _pre_seconds(self: CalibrationAnalyzer) -> float:
        """Return sum of lead-in and event pre-time."""
        return self.info.start - self.stream[0].stats.starttime

    def _post_seconds(self: CalibrationAnalyzer) -> float:
        """Return sum of lead-out and event post-time."""
        return self.stream[0].stats.endtime - self.info.end

    def _input_stream(self: CalibrationAnalyzer) -> Stream:
        return self.stream.select(component=CAL_COMPONENT)

    def _output_stream(self: CalibrationAnalyzer) -> Stream:
        input_stream_ids = [trace.id for trace in self._input_stream()]
        output_traces = [trace for trace in self.stream
                         if trace.id not in input_stream_ids]
        return Stream(output_traces)

    def load_response(self: CalibrationAnalyzer, pattern: str) -> None:
        """Load response file describing the system being calibrated."""
        response_files = {trace.id: '' for trace in self._output_stream()}
        self.logger.info(
            'Searching %s for: %s',
            pattern, ', '.join(sorted(response_files.keys())))
        for response_file in glob(pattern):
            inventory = read_inventory(response_file)
            for trace in self._output_stream():
                try:
                    trace.attach_response(inventory)
                except ValueError:
                    pass

            found_ids = (
                {trace.id for trace in self._output_stream()
                 if 'response' in trace.stats} -
                {trace_id for trace_id, file_name in response_files.items()
                 if file_name})

            if found_ids:
                self.logger.info(
                    '%s: %s', response_file, ', '.join(sorted(found_ids)))
                for trace_id in found_ids:
                    response_files[trace_id] = response_file

            if all('response' in trace.stats
                   for trace in self._output_stream()):
                break

        missing_traces = [trace for trace in self._output_stream()
                          if 'response' not in trace.stats]
        if missing_traces:
            missing_ids = ', '.join(trace.id for trace in missing_traces)
            self.logger.warning('No station metadata found: %s', missing_ids)

        for trace in missing_traces:
            self.stream.remove(trace)
        if not self._output_stream().traces:
            raise RuntimeError('No waveforms with station metadata')
        self.info.response_file = [
            response_files[trace.id] if trace.id in response_files else ''
            for trace in self.stream]

        sensor = self._sensor_stage()
        if sensor.input_units.upper() not in MOTION:
            self.logger.error(
                'Sensor input units %s not among supported: ',
                (sensor.input_units, ', '.join(sorted(MOTION.keys()))))

        if self._output_stream() and isinstance(self._output_stream()[0], Trace):
            first_trace = self._output_stream()[0]
            first_stats = first_trace.stats  # pylint: disable=no-member
            self.logger.debug(first_stats.response)

    def _force_response_match_deprecated(
        self: CalibrationAnalyzer,
        inventory: Inventory,
        stream: Stream,
    ) -> None:
        """Force inventory to have same NSLC codes as stream."""
        network_codes = {trace.stats.network for trace in stream}
        station_codes = {trace.stats.station for trace in stream}
        location_codes = {trace.stats.location for trace in stream}
        channel_codes = {trace.stats.channel for trace in stream}

        for network, network_code in zip(inventory, network_codes):
            network.code = network_code
            for station, station_code in zip(network, station_codes):
                station.code = station_code
                for channel, location_code, channel_code in zip(
                        station, location_codes, channel_codes):
                    channel.location_code = location_code
                    channel.code = channel_code

    def load_calibration_signal(
        self: CalibrationAnalyzer,
        calibration_signal_file: str,
    ) -> None:
        """
        Load calibration input signal.

        No attempt is made to match the response to the exact channel used;
        it is assumed that all traces in the calibation stream have the same
        nominal response.
        """
        if self._input_stream().traces:
            self.logger.warning(
                'Waveform data already includes calibration channel.')
            return

        self.logger.info(calibration_signal_file)
        self.info.calibration_signal_file = calibration_signal_file

        b_stages, factors = extract_decimation_coefficients(
            self.stream[0].stats.response.response_stages)
        factor = reduce(mul, factors)
        phase_suffix = ''
        if factors[-1] == 2:
            if len(b_stages[-1]) == 223:
                phase_suffix = '_linear'
            elif len(b_stages[-1]) == 110:
                phase_suffix = '_minimum'
        if not phase_suffix:
            self.logger.warning(
                'Cannot determine whether decimation filters are minimum or '
                'linear phase.')

        cache_file = os.path.join(
            gettempdir(),
            f'{calibration_signal_file.replace(".lzma", "")}_'
            f'{self._sampling_rate():g}sps{phase_suffix}_'
            f'delay{self.info.delay_start}s.{self.CACHE_FORMAT.lower()}')
        if os.path.isfile(cache_file):
            self.logger.info('Found cache: %s', cache_file)
            signal = read(cache_file)[0].data
        else:
            self.logger.info('Reading: %s', calibration_signal_file)
            signal = np.frombuffer(lzma.open(calibration_signal_file).read(),
                                   dtype=CALIBRATION_DTYPE)
            signal = sample_hold_digitize(signal)
            signal = pad_for_decimation(signal, b_stages, factors)[0]
            signal = self._pad_lead_in_out(signal, factor)

            self.logger.info('Decimating by %d ...', factor)
            signal = multi_decim(signal, b_stages, factors)[0]

        if len(signal) != self.stream[0].stats.npts:
            self.logger.error(
                'Expected %d, obtained %d points for calibration signal',
                self.stream[0].stats.npts, len(signal))
            raise RuntimeError('Check lead-in and lead-out times.')

        stats = {key: self.stream[0].stats[key] for key in
                 ['network', 'station', 'location', 'sampling_rate']}
        stats['channel'] = self.stream[0].stats['channel'][:2] + CAL_COMPONENT
        stats['starttime'] = self.stream[0].stats.starttime
        trace = Trace(data=np.ascontiguousarray(signal), header=stats)
        stream = Stream([trace])

        if not os.path.isfile(cache_file):
            self.logger.info('Caching: %s', cache_file)
            stream.write(cache_file, format=self.CACHE_FORMAT)

        self.stream += stream

    def check_stream(
        self: CalibrationAnalyzer,
        discard_s: float = DEFAULT_DISCARD,
    ) -> None:
        """
        Check for: clipping, gaps, misaligned start & end.

        An additional, small amount of data is discarded from start and end,
        before checking, required only for Guralp calibrations.
        """
        self.info.start += discard_s
        self.info.end -= discard_s

        last_start = max([trace.stats.starttime for trace in self.stream])
        if self.info.start < last_start:
            self.logger.warning(
                'Data missing, delaying start to %s', last_start)
            self.info.start = last_start + discard_s

        first_end = min([trace.stats.endtime for trace in self.stream])
        if self.info.end < first_end:
            self.logger.warning(
                'Data missing, advancing end to %s', first_end)
            self.info.end = first_end - discard_s
        self.logger.debug(str(self.stream))

        if self.stream.get_gaps():
            with StringIO() as buffer, redirect_stdout(buffer):
                self.stream.print_gaps()
                self.logger.warning(buffer.getvalue())
            raise RuntimeError('Cannot process waveforms with gaps.')

        for trace in self._output_stream():
            clipping = np.abs(trace.data) > CHECK_CLIP
            if any(clipping):
                i = np.argmax(clipping)
                self.logger.warning(
                    'Potential clipping (|%d| > %d) counts on %s at %s',
                    trace.data[i], CHECK_CLIP, trace.id,
                    trace.stats.starttime + i*trace.stats.delta)

    def _pad_lead_in_out(
        self: CalibrationAnalyzer,
        samples: ArrayLike,
        factor: float,
    ) -> np.ndarray:
        sampling_rate = self._sampling_rate()*factor
        return np.hstack((
            np.zeros(int(round(self._pre_seconds()*sampling_rate))),
            samples,
            np.zeros(int(round(self._post_seconds()*sampling_rate)))))

    def _sensor_stage(self: CalibrationAnalyzer) -> ResponseStage:
        """
        Return sensor response stage.

        Note assumptions:
            1. Sensor is a single stage.
            2. All traces have the same response.
        """
        return self._output_stream()[0].stats.response.response_stages[0]  # pylint: disable=no-member

    def _cal_stage(self: CalibrationAnalyzer) -> ResponseStage:
        """
        Return calibration response stage.

        Note assumptions:
            1. Sensor is a single stage.
            2. All traces have the same response.
        """
        return self._input_stream()[0].stats.response.response_stages[0]

    def sensitivity(
        self: CalibrationAnalyzer,
        system: lti
    ) -> Tuple[float, float]:
        """Compute sensitivity at sensor stage gain frequency."""
        f_norm = self._sensor_stage().stage_gain_frequency
        sensitivity = np.abs(system.freqresp(w=2*np.pi*f_norm)[1][0])
        return f_norm, sensitivity

    def load_calibration_response(
        self: CalibrationAnalyzer,
        response_file: str = '',
    ) -> None:
        """Set up calibration input response."""
        self.logger.info(response_file)
        inventory = read_inventory(response_file)
        cal_inventory = inventory.select(
            network=self._input_stream()[0].stats.network,
            station=self._input_stream()[0].stats.station,
            location=self._input_stream()[0].stats.location,
            channel=self._input_stream()[0].stats.channel,
            time=self._input_stream()[0].stats.starttime,
        )
        if cal_inventory:
            response = cal_inventory[0][0][0].response
        else:
            response = inventory[0][0][0].response
        instrument_sensitivity = response.instrument_sensitivity.value
        # n.b. can't use response.recalculate_overall_sensitivity() because it
        # takes the absolute value!
        response.instrument_sensitivity.value = np.prod([
            stage.stage_gain for stage in response.response_stages])

        if not np.isclose(response.instrument_sensitivity.value,
                          instrument_sensitivity, rtol=CHECK_PERCENT/100):
            units = (
                f'{response.instrument_sensitivity.output_units}/'
                f'({response.instrument_sensitivity.input_units.lower()})')
            self.logger.warning(
                'Instrument sensitivity %.6g %s in file differs from'
                'recalculated value %.6g %s by more than %g%%.',
                instrument_sensitivity, units,
                response.instrument_sensitivity.value, units, CHECK_PERCENT)

        recompute_normalization_factors(response, rtol=CHECK_PERCENT/100)

        if MOTION[response.response_stages[0].input_units.upper()] != 'ACC':
            self.logger.warning(
                'Expected calibration input [%s] to be acceleration [%s]: ',
                response.response_stages[0].input_unit, UNITS['ACC'])

        decimation_stages = [
            stage for stage in self.stream[0].stats.response.response_stages
            if stage.decimation_factor and stage.decimation_factor > 1]
        input_stages = [
            stage for stage in response.response_stages
            if not stage.decimation_factor or stage.decimation_factor == 1]

        self._input_stream()[0].stats.response = Response(
            response_stages=input_stages + decimation_stages,
            instrument_sensitivity=response.instrument_sensitivity)
        for i, stage in enumerate(
                self._input_stream()[0].stats.response.response_stages,
                start=1):
            stage.stage_sequence_number = i

        self.info.calibration_response_file = response_file
        self.logger.debug(str(self._input_stream()[0].stats.response))

    def setup_nominal_responses(self: CalibrationAnalyzer) -> None:
        """
        Set up nominal sensor, cal & system (linear time-invariant) responses.

        system = calibration [m/s^2/V] * sensor [V/(m/s^n)] * integrator(2 - n)

        Note that calibration response (as recorded in the station metadata) is
        the conversion from ground motion to voltage, so to convert voltage to
        ground motion it is inverted.
        """
        sensor = self._sensor_stage()
        self.lti.sensor = zpk_from_zpsf(
            sensor.zeros, sensor.poles,
            sensor.stage_gain, sensor.normalization_frequency)
        self.logger.debug('Sensor: %s', str(self.lti.sensor))

        cal = self._cal_stage()
        cal_zeros, cal_poles, cal_gain = cal.poles, cal.zeros, 1/cal.stage_gain
        integrations = (ORDER[MOTION[sensor.input_units.upper()]] -
                        ORDER[MOTION[cal.input_units.upper()]])
        if integrations:
            self.logger.debug(
                'Integrating %d times, from %s to %s',
                integrations, cal.input_units, sensor.input_units)
            cal_poles = np.array([0]*integrations + list(cal_poles))
            # TODO: verify minus sign - is this only Trillium 120?
            cal_gain /= (-2*np.pi*cal.normalization_frequency)**integrations
        self.lti.cal = zpk_from_zpsf(
            cal_zeros, cal_poles, cal_gain, cal.normalization_frequency)
        self.logger.debug('Cal: %s', self.lti.cal)

        system_zeros = (self.lti.cal.zeros.tolist() +
                        self.lti.sensor.zeros.tolist())
        system_poles = (self.lti.cal.poles.tolist() +
                        self.lti.sensor.poles.tolist())
        system_gain = self.lti.cal.gain*self.lti.sensor.gain
        self.lti.system = zpk_cancel(
            lti(system_zeros, system_poles, system_gain))
        self.logger.debug('System: %s', self.lti.system)

    def map_orientations(self: CalibrationAnalyzer, mapping) -> None:
        """Account for differences between outputs and internal mechanics."""
        for trace in self.stream:
            for output, internal in zip(*mapping):
                if trace.id[-1] == output:
                    trace.id = trace.id[:-1] + internal

    def tf_nominal(
        self: CalibrationAnalyzer,
        model: str,
        f: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """
        Return nominal transfer function evaluated at frequencies.

        Arguments:
        - `model`: typically 'sensor', 'cal' or 'system'
        - `f`: frequencies in Hz, defaults to those of self.stft

        Returns:
        - `tf`: nominal transfer function at given frequencies.
        """
        if model not in vars(self.lti):
            raise ValueError(
                f"Model '{model}' not among supported: "
                ', '.join(vars(self.lti)))
        if f is None:
            f = self.stft.f
        else:
            f = np.array(f)

        omegas = 2*np.pi*f
        tf_sensor = self.lti.sensor.freqresp(omegas)[1]
        tf_cal = self.lti.cal.freqresp(omegas)[1]

        if model == 'cal':
            return tf_cal
        elif model == 'sensor':
            return tf_sensor
        else:
            return tf_sensor*tf_cal

    def voltage(
        self: CalibrationAnalyzer,
        which: str = 'output',
        trim: bool = True,
    ) -> np.ndarray:
        """
        Get digitizer input or output signals in volts.

        Arguments:
        - `which`: "input" or "output"
        - `trim`: remove padding (e.g. turn-on and turn-off times) by slicing

        Returns:
        - `signal`: one row per channel
        """
        assert which in ['input', 'output']

        if which == 'input':
            stream = self._input_stream()
        else:
            stream = self._output_stream()

        if trim:
            stream = stream.slice(self.info.start, self.info.end)
        signal = np.row_stack([trace.data for trace in stream]).astype(float)

        digitizer_sensitivity = np.array([
            trace.stats.response.instrument_sensitivity.value /
            trace.stats.response.response_stages[0].stage_gain
            for trace in stream]).reshape((-1, 1))
        self.logger.debug(
            'Digitizer sensitivities: %s', digitizer_sensitivity.squeeze())
        signal /= digitizer_sensitivity

        return signal

    def compute(self: CalibrationAnalyzer, len_fft=None, num_windows=30, fraction_overlap=0.5,
                window=DEFAULT_WINDOW):
        """
        Time-consuming part of calibration analysis is done here.

        Each segment is detrended by removing a constant value before
        application of a 'hanning' window.

        Analysis is performed on sensor input and output signals in volts.
        """
        x = self.voltage('input', trim=True)
        y = self.voltage('output', trim=True)

        f_sample = self._sampling_rate()
        num_samples = x.shape[1]

        if len_fft is None:
            len_fft = len_fft_welch(num_samples, num_windows, fraction_overlap)
        len_overlap = int(fraction_overlap*len_fft)

        num_windows = num_windows_welch(num_samples, len_fft, len_overlap)

        self.stft.compute(x, y, f_sample, len_fft, len_overlap, window)
        self.stft.trim(low_frequency_points=2)

    def summary(self: CalibrationAnalyzer):
        """
        Summarize calibration result in a table, one row per trace.

        Columns are pd.MultiIndex.
            First level:
                FAP: 'info', 'magnitude_db', 'phase_deg', 'variance_db'
                PAZ: 'info', 'zeros', 'poles', 'gain'

            Second Level (except for info):
                FAP: frequency
                PAZ: pole/zero index

        Rows are indexed by obspy.Trace.id and start time.
        """
        # first construct info for calibration outputs
        info = pd.DataFrame(index=pd.MultiIndex.from_product(
            [[pd.to_datetime(self.info.start.datetime)],
             [trace.id for trace in self._output_stream()]],
            names=['start', 'trace_id']))

        for key, value in pd.Series(vars(self.info)).drop(
                ['start', 'end', 'calibration_signal_file',
                 'calibration_response_file']).items():
            try:
                info[key] = value
            except ValueError:
                info[key] = value[:info.shape[0]]
        info['duration_s'] = self.info.end - self.info.start
        info['pre_s'] = self._pre_seconds()
        info['post_s'] = self._post_seconds()
        info['sampling_rate_sps'] = self._sampling_rate()
        info['windows'] = self.stft.num_windows()
        info['windows'] = info['windows'].astype(int)
        if self.timing_gain_fit.timing is not None:
            info['timing error estimate [s]'] = round(self.timing_gain_fit.timing.params[0], 6)
            info['timing error uncertainty [s]'] = round(
                self.timing_gain_fit.num_sigma()*self.timing_gain_fit.timing.bse[0], 6)
        if self.timing_gain_fit.gain is not None:
            info['gain error estimate [%]'] = round(
                100*(self.timing_gain_fit.gain.params[0] - 1), 4)
            info['gain error uncertainty [%]'] = round(
                100*(self.timing_gain_fit.num_sigma()*self.timing_gain_fit.gain.bse[0]), 4)
        info['confidence [%]'] = self.timing_gain_fit.confidence

        # now append row for calibration input
        info = pd.concat((info, pd.Series(
            name=(pd.NaT, self._input_stream()[0].id),
            dtype=float).to_frame().T))
        info.loc[info.index[-1], 'waveform_file'] = \
            self.info.calibration_signal_file
        info.loc[info.index[-1], 'response_file'] = \
            self.info.calibration_response_file

        # finally add info relating to all rows
        info['output_units'] = self._sensor_stage().output_units
        info['input_units'] = self._sensor_stage().input_units.lower()
        info[PACKAGE] = VERSION
        units = (f'{self._sensor_stage().output_units}/'
                 f'({self._sensor_stage().input_units.lower()})')
        if self.lti.fits:
            poles = pd.concat(
                [feature_stats(zpk_divide(fit, self.lti.cal).poles, 'pole')
                    for fit in self.lti.fits] +
                [feature_stats(self.lti.sensor.poles, 'pole')]
            ).applymap(lambda x: round_sig(x, 6))
            poles.index = info.index

            zeros = pd.concat(
                [feature_stats(zpk_divide(fit, self.lti.cal).zeros, 'zero')
                    for fit in self.lti.fits] +
                [feature_stats(self.lti.sensor.zeros, 'zero')]
            ).applymap(lambda x: round_sig(x, 6))
            zeros.index = info.index

            sensitivity = pd.DataFrame(
                np.vstack(
                    [self.sensitivity(zpk_divide(fit, self.lti.cal))
                     for fit in self.lti.fits] +
                    [self.sensitivity(self.lti.sensor)]),
                columns=pd.MultiIndex.from_tuples([('f_norm', 'hz'),
                                                   ('sensitivity', units)]),
                index=info.index).applymap(lambda x: round_sig(x, 6))

            info.columns = pd.MultiIndex.from_product((['info'], info.columns))

            df = pd.concat((info, poles, zeros, sensitivity), axis=1)

        else:
            f = pd.Index(self.stft.f, name='f')
            tf_estimate = np.divide(self.stft.tf_estimate(),
                                    self.tf_nominal('cal'))
            tf_nominal = self.tf_nominal('sensor')

            magnitude = pd.DataFrame(
                np.vstack((20*np.log10(np.abs(tf_estimate)),
                           20*np.log10(np.abs(tf_nominal)))).round(3),
                columns=f, index=info.index)
            phase = pd.DataFrame(
                np.vstack((np.angle(tf_estimate, deg=True),
                           np.angle(tf_nominal, deg=True))).round(3),
                columns=f, index=info.index)
            variance = pd.DataFrame(
                np.vstack((20*np.log10(self.stft.variance()),
                           np.full_like(f.values, np.NaN))).round(3),
                columns=f, index=info.index)

            df = pd.concat(
                (info, magnitude, phase, variance),
                keys=['info', 'magnitude_db', 'phase_deg', 'variance_db'],
                axis=1)

        df.columns.names = ['type', 'subtype']

        return df

    def tf_fits(self: CalibrationAnalyzer) -> np.ndarray:
        """Compute fitted transfer functions at measured frequencies."""
        return np.array([
            fit.freqresp(2*np.pi*self.stft.f)[1] for fit in self.lti.fits])

    def test(
        self: CalibrationAnalyzer,
        test_band_hz: Tuple[float, float] = TEST_BAND_HZ,
        max_amplitude_percent: float = MAX_AMPLITUDE_PERCENT,
        max_phase_degrees: float = MAX_PHASE_DEGREES,
    ) -> None:
        """
        Check whether measured transfer function deviation is within spec.

        Specifciation is defined with respect to nominal.

        Arguments:
        - `test_band_hz`: minimum and maxumum frequency of interest
        - `max_amplitude_percent`: maximum percentage deviation of amplitude
        - `max_phase_degrees`: maximum deviation of phase in degrees
        """
        try:
            tf_estimate = self.tf_fits()
        except AttributeError:
            tf_estimate = self.stft.tf_estimate()
        tf_nominal = self.tf_nominal('system').reshape((1, -1))
        tf_deviation = tf_estimate/tf_nominal

        self.info.spec_min_freq_hz = test_band_hz[0]
        self.info.spec_max_freq_hz = test_band_hz[1]
        self.info.spec_max_amp_pct = max_amplitude_percent
        self.info.spec_max_phase_deg = max_phase_degrees

        self.info.in_spec = ~np.any(
            ((np.abs(100*(np.abs(tf_deviation) - 1)) >
              self.info.spec_max_amp_pct) |
             (np.abs(np.angle(tf_deviation, deg=True)) >
              self.info.spec_max_phase_deg)) &
            ((self.info.spec_min_freq_hz <= self.stft.f) &
             (self.stft.f <= self.info.spec_max_freq_hz)), axis=1)

        self.logger.info('Result: %s', ', '.join(
            [': '.join(items) for items in zip(
                [trace.id[-1] for trace in self.stream],
                ['pass' if in_spec else 'fail'
                 for in_spec in self.info.in_spec])]))

    def write_calibrate_result(
        self: CalibrationAnalyzer,
        ims_instrument_type: str = ''
    ) -> None:
        """Write IMS2.0 CALIBRATE_RESULT message with CAL2 and FAP2."""
        keep = ((self.stft.f >= self.info.spec_min_freq_hz) &
                (self.stft.f <= self.info.spec_max_freq_hz))
        f = self.stft.f[keep]
        if len(f) >= MAX_FAP_LEN:
            self.logger.warning(
                '%d frequencies is more than %d supported by FAP2 format',
                len(f), MAX_FAP_LEN)
        tf_estimate = (
            self.stft.tf_estimate()[:, keep] /
            self.tf_nominal('cal')[keep])

        output_txt = '_'.join([
            'calibrate_result', self.stream[0].stats.station,
            self.info.start.strftime('%Y%m%d.%H%M')]) + '.txt'
        self.logger.info(output_txt)

        with open(output_txt, 'w', encoding='UTF-8') as file:
            file.write(IMS_HEADER.format(
                msg_id=CAL_RESULT_MSG_ID.format(
                    year=self.info.start.year,
                    station=self.stream[0].stats.station),
                ref_id=CAL_RESULT_REF_ID.format(
                    year=self.info.start.year,
                    station=self.stream[0].stats.station),
                time_stamp=UTCDateTime().strftime(IMS_DATETIME_FMT)))

            for trace, amplitudes, phases, in_spec in zip(
                    self.stream, np.abs(tf_estimate),
                    np.angle(tf_estimate, deg=True),
                    self.info.in_spec):

                response = trace.stats.response
                assert (response.instrument_sensitivity.frequency ==
                        response.response_stages[0].stage_gain_frequency)
                assert response.instrument_sensitivity.output_units == 'COUNTS'
                calper = 1/self._sensor_stage().stage_gain_frequency
                nominal_sensor = self._sensor_stage().stage_gain
                nominal_instrument = response.instrument_sensitivity.value
                nominal_digitizer = nominal_instrument/nominal_sensor
                actual_sensor = amplitudes[np.argmax(f >= 1/calper)]
                calib = 1e9*calper/(2*np.pi*actual_sensor*nominal_digitizer)
                file.write(RESPONSE_HEADER.format(
                    station=trace.stats.station,
                    channel=trace.stats.channel,
                    calib=calib,
                    calper=calper,
                    in_spec='YES' if in_spec else 'NO'))
                file.write(CAL_BLOCK.format(
                    station=trace.stats.station,
                    channel=trace.stats.channel,
                    aux_id='',
                    inst_type=ims_instrument_type,
                    calib=calib,
                    calper=calper,
                    sample_rate=self._sampling_rate(),
                    start=self.info.start.strftime(IMS_DATETIME_FMT),
                    end=self.info.end.strftime(IMS_DATETIME_FMT)))
                file.write(FAP_HEADER.format(
                    stage=1,
                    units=self._sensor_stage().output_units,
                    decimation='',
                    group_correction=0,
                    count=len(f),
                    description=('Input units: ' +
                                 self._sensor_stage().input_units.lower())))
                for frequency, amplitude, phase in zip(
                        f, amplitudes, phases):
                    file.write(FAP_DATA.format(
                        frequency=frequency,
                        amplitude=amplitude,
                        phase=phase))

            file.write(IMS_FOOTER)

    def simulate_response(
        self: CalibrationAnalyzer,
        trim: bool = True,
        model: str = 'system',
    ) -> np.ndarray:
        """Simulate nominal response of sensor to calibration signal."""
        zpk = getattr(self.lti, model)
        sensitivity = self.sensitivity(zpk)[1]
        nominal_paz = {'zeros': zpk.zeros,
                       'poles': zpk.poles,
                       'gain': zpk.gain/sensitivity,
                       'sensitivity': sensitivity}
        signal = simulate_seismometer(
            self.voltage('input', trim=False).squeeze(), self._sampling_rate(),
            paz_simulate=nominal_paz, simulate_sensitivity=True)

        if trim:
            num_start = int(self._pre_seconds()*self._sampling_rate())
            num_end = int(self._post_seconds()*self._sampling_rate())
            signal = signal[num_start:-num_end]

        return signal

    def fit(self: CalibrationAnalyzer) -> None:
        """Least-squares estimation of poles and zeros."""
        f = self.stft.f
        if len(f) < 1:
            self.logger.error('No data to fit.')
            return

        tf_estimates = self.stft.tf_estimate()
        zpk_nom = self.lti.system
        variances = self.stft.variance()
        self.logger.debug(zpk_nom)

        f_norm, sens_nom = self.sensitivity(self.lti.sensor)

        zpk_fixed = zpk_out_of_band(self.lti.system, self.stft.f,
                                    self._sensor_stage().stage_gain_frequency)
        self.logger.debug(zpk_fixed)

        units = (f'{self._sensor_stage().output_units}/'
                 f'({self._sensor_stage().input_units.lower()})')
        zpk_unfixed = zpk_divide(zpk_nom, zpk_fixed)
        self.logger.info(
            'Nominal zeros [rad/s]: %s', feature_str(zpk_unfixed.zeros))
        self.logger.info(
            'Nominal poles [rad/s]: %s', feature_str(zpk_unfixed.poles))
        self.logger.info(
            'Nominal sensitivity [%s at %g Hz]: %.5g', units, f_norm, sens_nom)

        self.lti.fits = []
        self.zpk_fits = []
        labels = factor_names(self.stream)[1]
        for label, tf_estimate, variance in zip(labels, tf_estimates, variances):

            self.logger.info('Fitting: %s', label)
            zpk_fit = fit_response(
                zpk_nom, f, tf_estimate, variance, zpk_fixed, debug=True)

            self.logger.debug(zpk_fit)
            zpk_unfixed = zpk_divide(zpk_fit, zpk_fixed)
            f_norm, sens_fit = self.sensitivity(
                zpk_divide(zpk_fit, self.lti.cal))
            self.logger.info(
                'Fit zeros [rad/s]: %s', feature_str(zpk_unfixed.zeros))
            self.logger.info(
                'Fit poles [rad/s]: %s', feature_str(zpk_unfixed.poles))
            self.logger.info(
                'Fit sensitivity [%s at %g Hz]: %.5g', units, f_norm, sens_fit)

            self.lti.fits.append(zpk_fit)
            # self.zpk_fits.append(result)

    def estimate_errors(
        self: CalibrationAnalyzer,
        variance_threshhold: float = 0.03
    ) -> None:
        """
        Least-squares estimation of gain and timing errors.

        The optimal estimate of the error delta_gain is simply the average
        over frequency weighted by the estimated variance at each frequency.
        Alternately we have B=gain [Mx1] and desire to find x=gain_error
        [scalar] which is optimal over all frequencies so we set A=1 [Mx1]
        and use readily available tools to minimize the Euclidean 2-norm
        ||B - A*x||^2, a standard problem of linear algebra.

        The random error in the estimated gain is
        (Bendat & Piersol, 1993, eq. 11.55, p. 307):
            gain_variance = (1/gamma^2 - 1)/(2*n)
        Where gamma is the coherence and n is the number of statistically
        independent windows. The optimal weight is the inverse square root of
        the variance.

        A time delay between input and output of delta_t [s] results in an
        apparent phase shift, delta_theta [radians] which varies linearly
        with angular frequency w [radians/s]:
            delta_theta = delta_t * w
        We have m measurements of the phase shift B=delta_theta [Mx1 radians]
        at each frequency A=w [Mx1 radians/s], and we want to find
        x=delta_t [scalar s] which minimizes the Euclidean 2-norm
        ||B - A*x||^2, a standard problem of linear algebra.

        When the gain variance is sufficiently small, the random error in the
        expected phase, measured in radians, is actually the same as for the
        gain_variance (Bendat & Piersol, 1993, eq. 11.58, p. 309)
            phase_variance = (1/gamma^2 - 1)/(2*n) [radians]
        This is the optimal weight for a least-squares solution.

        Although this procedure could be modified to compute estimates of
        errors on a per-channel basis, it instead treats the errors as common
        to all channels.

        Arguments:
        - `variance_threshhold`: maximum variance for inclusion in fit
        """
        f = self.stft.f
        tf_estimate = self.stft.tf_estimate()
        with np.errstate(divide='ignore', invalid='ignore'):
            tf_estimate /= self.tf_nominal('system')

        variance = self.stft.variance()
        if variance_threshhold:
            keep = (variance < variance_threshhold).any(axis=0)
        else:
            keep = np.full_like(variance, True)

        tf_estimate = tf_estimate[:, keep]
        variance = variance[:, keep]
        f = f[keep]
        if not all(keep):
            self.logger.info(
                'Discarding %d/%d points with variance > %g',
                (~keep).sum(), len(keep), variance_threshhold)

        if len(f) < 1:
            raise RuntimeError('No data to fit.')

        magnitude = np.abs(tf_estimate)
        phase = unwrap_mid(np.angle(tf_estimate), f, axis=1)

        f = np.reshape(np.tile(f, magnitude.shape[0]), (-1, 1))
        magnitude = np.reshape(magnitude, (-1, 1))
        phase = np.reshape(phase, (-1, 1))
        variance = np.reshape(variance, (-1, 1))
        weights = np.sqrt(1/variance)

        self.timing_gain_fit.gain = sm.WLS(magnitude, np.ones_like(f), weights).fit()
        self.logger.info(self.timing_gain_fit.gain_summary())
        self.timing_gain_fit.timing = sm.WLS(phase, 2*np.pi*f, weights).fit()
        self.logger.info(self.timing_gain_fit.timing_summary())

    def _save_image(self: CalibrationAnalyzer, fig, option_list=None):
        """Save a figure with an automatically descriptive file name."""
        if not self.savefig or not self.dpi:
            return

        if isinstance(option_list, str):
            option_list = option_list.split(',')

        # pylint:disable=protected-access
        caller_name = sys._getframe(1).f_code.co_name
        plot_type = caller_name.replace('plot_', '')

        file_parts = [plot_type]

        if option_list is not None:
            file_parts += [option for option in option_list if option]

        if self.stream is not None:
            start_string = np.max(
                [trace.stats.starttime for trace in self.stream]
            ).strftime('%Y%m%d.%H%M')

            common_name = factor_names(self.stream)[0]

            file_parts += [common_name, start_string]

        output_png = '_'.join(file_parts) + '.png'

        self.logger.info(output_png)
        fig.savefig(output_png, dpi=self.dpi, bbox_inches='tight')

    def plot_check(self: CalibrationAnalyzer, where='start', window_seconds=10):
        """Spot check critical times in the calibration."""
        assert where in ['start', 'end', 'on', 'off']

        if where == 'on':
            target_time = self.stream[0].stats.starttime + window_seconds/2
        elif where == 'off':
            target_time = self.stream[0].stats.endtime - window_seconds/2
        else:
            target_time = getattr(self.info, where)

        stream = self.stream.slice(target_time - window_seconds/2,
                                   target_time + window_seconds/2)
        fig = stream.plot(handle=True, equal_scale=False)
        if where in ['start', 'end']:
            for ax in fig.axes:
                ax.axvline(target_time.datetime, linestyle='--', linewidth=0.5,
                           label=where)
        fig.axes[-1].legend(loc='lower left')

        self._save_image(fig, option_list=where)

    ALLOWED_NOMINAL_MODEL_REMOVALS = ['system', 'cal']

    def plot_response(self: CalibrationAnalyzer, model='system', f_limits=None):
        """
        Plot nominal transfer function between specified frequency limits.

        Arguments:
        - `model`:
            - 'sensor' for the sensor itself,
            - 'cal' for calibration input or
            - 'system' for the combination of the two
        """
        if f_limits:
            f = logspace(f_limits[0], f_limits[1], 24)
        else:
            f = self.stft.f
        tf_nominal = self.tf_nominal(model, f=f)
        gain_db = 20*np.log10(np.abs(tf_nominal))
        phase_deg = np.angle(tf_nominal, deg=True)

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width, width))
        axes[0].semilogx(f, gain_db, label=model)
        axes[1].semilogx(f, phase_deg, label=model)
        axes[0].axvline(self._sampling_rate()/2, linestyle='--', color='0.5',
                        label='Nyquist')
        axes[1].axvline(self._sampling_rate()/2, linestyle='--', color='0.5',
                        label='Nyquist')
        if np.any(np.abs(axes[1].get_ylim()) > 180):
            axes[1].set_ylim([-180, 180])
            axes[1].set_yticks(np.arange(-180, 180 + 1, 45.))

        if model == 'cal':
            output_units = self._sensor_stage().input_units.lower()
            input_units = self._sensor_stage().output_units
            axes[0].set_ylabel(f'Gain [dB wrt {output_units}/({input_units})]')
        elif model == 'sensor':
            output_units = self._sensor_stage().output_units
            input_units = self._sensor_stage().input_units.lower()
            axes[0].set_ylabel(f'Gain [dB wrt {output_units}/({input_units})]')
        else:
            axes[0].set_ylabel('Gain [dB]')
        axes[0].legend(loc='best')
        axes[1].set_ylabel('Phase [°]')
        axes[1].set_xlabel('Frequency [Hz]')
        subplots_squeeze(fig, hspace=0)

        self._save_image(fig, model)

    def plot_simulated(self: CalibrationAnalyzer, trim=True):
        """Plot simulated calibration response in time domain."""
        simulated = self.simulate_response(trim=trim)

        fig, ax = plt.subplots()
        if self.stream is not None:
            labels = factor_names(self.stream)[1]
            for output, label in zip(self.voltage('output', trim=trim),
                                     labels):
                ax.plot(np.arange(len(output))/self._sampling_rate(), output,
                        label=label)
        ax.plot(np.arange(len(simulated))/self._sampling_rate(), simulated,
                label='simulated')
        ax.set_ylabel('Voltage [V]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper left')

        self._save_image(fig)

    def plot_signal_to_noise(self: CalibrationAnalyzer):
        """Plot estimated signal-to-noise ratio."""
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]
        coherence_squared = self.stft.coherence_squared()

        fig, ax = plt.subplots()
        for snr, label in zip(10*np.log10(1/(1 - coherence_squared)), labels):
            ax.semilogx(self.stft.f, snr, label=label)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('SNR [dB]')
        ax.legend(loc='upper left')

        self._save_image(fig)

    def plot_spectrogram(self: CalibrationAnalyzer):
        """Plot input and output spectrograms, referred to sensor input [V]."""
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]

        # convert to volts and make stackable
        f = self.stft.f
        t = self.stft.t
        tf_nominal = self.tf_nominal('system')
        tf_nominal = tf_nominal.reshape((1, -1, 1))
        p_dbs = 10*np.log10(np.concatenate(
            (self.stft.p_xx.reshape(-1, len(f), len(t)),
             self.stft.p_yy/np.abs(tf_nominal)**2), axis=0))
        max_db = p_dbs.max()

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(len(labels), 1, sharex=True,
                                 figsize=(width, len(labels)*width/3))
        for p_db, ax, label in zip(p_dbs, axes, labels):

            image = ax.pcolormesh(t, f, p_db, vmin=max_db - 100, vmax=max_db)
            ax.set_ylabel('Frequency [Hz]')
            ax.text(0.95, 0.9, label, transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(facecolor='white'))

        axes[-1].set_xlabel('Time [s]')
        subplots_squeeze(fig, hspace=0)

        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7,
                     label='Power Spectral Density [dB wrt $V^2/Hz$]')

        self._save_image(fig)

    def plot_variance(self: CalibrationAnalyzer, scale='log'):
        """
        Plot variance on log or linear scale.

        Arguments:
        - `scale` selects 'log' or 'linear' scaling for x-axis
        """
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        labels = factor_names(self.stream)[1]

        variances = self.stft.variance()

        fig, ax = plt.subplots()
        for snr, label in zip(variances, labels):
            ax.semilogx(self.stft.f, snr, label=label)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Relative Transfer Function Variance')
        ax.set_yscale(scale)
        ax.legend(loc='upper left')

        self._save_image(fig)

    def plot_transfer_function(
        self: CalibrationAnalyzer,
        remove: str = 'system',
        errors: str = 'estimate',
        scale: str = 'log',
        variance_threshhold: float = np.NaN,
    ) -> None:
        """
        Plot calibration transfer function on log or linear scale.

        Arguments:
        - `scale`: scaling for x-axis
        - `remove`: which nominal response to remove
            - '': no response removal
            - 'system': remove nominal system resposne (default)
            - 'cal': remove nominal calibration input response
        - `errors` : how to treat gain and phase errors
            - '': no estimation or correction
            - 'estimate': estimate only (default)
            - 'correct': estimate and correct
        - `variance_threshhold`: plot only points with variance above this
        """
        if remove and remove not in ['cal', 'system']:
            raise ValueError(
                f"Removal of '{remove}' response not supported")
        if errors and errors not in ['estimate', 'correct']:
            raise ValueError(f"Cannot '{errors}' errors")
        if self.stft.f is None:
            raise RuntimeError('Use compute() method first.')

        option_list = []
        if remove:
            option_list.append('nominal_' + remove + '_removed')

        channels = factor_names(self._output_stream())[1]
        results = ['pass' if item else 'fail' for item in self.info.in_spec]
        labels = [f'{channel}: {result}' for channel, result in zip(channels, results)]

        f = self.stft.f
        tf_nominal = self.tf_nominal('system')
        tf_estimate = self.stft.tf_estimate()
        variance = self.stft.variance()

        if self.lti.fits:
            tf_fits = self.tf_fits()

        if remove:
            tf_remove = self.tf_nominal(remove)
            with np.errstate(divide='ignore', invalid='ignore'):
                tf_estimate /= tf_remove
                tf_nominal /= tf_remove
                if self.lti.fits:
                    tf_fits /= tf_remove

        if errors:
            if self.timing_gain_fit.gain is None or self.timing_gain_fit.timing is None:
                raise RuntimeError('Run estimate_errors() first.')

            tf_error = tf_nominal*np.exp(1j*2*np.pi*f*self.timing_gain_fit.timing.params)
            tf_error *= self.timing_gain_fit.gain.params
        if errors == 'correct' and self.timing_gain_fit.gain is not None and \
                self.timing_gain_fit.timing is not None:
            tf_timing_gain = self.timing_gain_fit.gain.params*np.exp(
                1j*2*np.pi*f*self.timing_gain_fit.timing.params)
            with np.errstate(divide='ignore', invalid='ignore'):
                tf_estimate /= tf_timing_gain
                if self.lti.fits:
                    tf_fits /= tf_timing_gain

        if not np.isnan(variance_threshhold):
            tf_estimate[variance > variance_threshhold] = np.nan
            option_list += [f'variance_lt_{variance_threshhold}']

        band_hz = (self.info.spec_min_freq_hz,
                   self.info.spec_max_freq_hz)
        spec = (f >= band_hz[0]) & (f <= band_hz[1])
        gain_estimate = 20*np.log10(np.abs(tf_estimate))
        gain_nominal = 20*np.log10(np.abs(tf_nominal))
        gain_spec = np.vstack((gain_estimate, gain_nominal))[:, spec]
        phase_estimate = np.angle(tf_estimate, deg=True)
        phase_nominal = np.angle(tf_nominal, deg=True)
        phase_spec = np.vstack((phase_estimate, phase_nominal))[:, spec]

        width = plt.rcParams['figure.figsize'][0]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(width, width))

        for gain, label in zip(gain_estimate, labels):
            axes[0].plot(f, gain, label=label)
        axes[0].set_xlim((f[1], f[-1]))
        if not np.allclose(gain_nominal, 0):
            axes[0].plot(f, gain_nominal, label='nominal')
        axes[0].set_ylim((floor(gain_spec.min()) - 1,
                          ceil(gain_spec.max()) + 1))

        if errors == 'estimate':
            axes[0].plot(f, 20*np.log10(np.abs(tf_error)), label='error')

        for phase, label in zip(phase_estimate, labels):
            axes[1].plot(f, phase, label=label)

        axes[1].set_xlabel('Frequency [Hz]')

        if not np.allclose(phase_nominal, 0):
            axes[1].plot(f, unwrap_mid(phase_nominal, f, discont=180),
                         label='nominal')
        axes[1].set_ylim((floor(phase_spec.min()) - 10,
                          ceil(phase_spec.max()) + 10))
        if errors == 'estimate':
            axes[1].plot(f, np.angle(tf_error, deg=True), label='error')

        for tf_fit, label in zip(tf_fits, labels):
            gain_fit = 20*np.log10(np.abs(tf_fit))
            phase_fit = np.angle(tf_fit, deg=True)
            axes[0].plot(f, gain_fit, label=label)
            axes[1].plot(f, phase_fit, label=label)

        max_mag_db = 20*np.log10((1 + self.info.spec_max_amp_pct/100))
        max_phase_deg = self.info.spec_max_phase_deg
        ids_start = '\n'.join([
            factor_names(self.stream)[0],
            self.info.start.strftime('%Y-%m-%d %H:%M')])
        test_limits = (
            f'±{self.info.spec_max_amp_pct}%, '
            f'±{self.info.spec_max_phase_deg}°\n'
            f'{band_hz[0]} - {band_hz[1]} Hz')
        axes[0].annotate(ids_start, (0.025, 0.025), xycoords='axes fraction',
                         ha='left', va='bottom')
        axes[0].fill_between(f[spec],
                             gain_nominal[spec] - max_mag_db,
                             gain_nominal[spec] + max_mag_db,
                             color='0.5', alpha=0.5, label=test_limits)
        axes[1].fill_between(f[spec],
                             phase_nominal[spec] - max_phase_deg,
                             phase_nominal[spec] + max_phase_deg,
                             color='0.5', alpha=0.5, label=test_limits)

        if remove == 'system':
            axes[0].set_ylabel('Gain wrt nominal [dB]')
        elif remove == 'cal':
            axes[0].set_ylabel(
                f'Gain [dB wrt {self._sensor_stage().output_units}/'
                f'({self._sensor_stage().input_units.lower()})]')
        else:
            axes[0].set_ylabel('Gain [dB]')
        axes[1].set_ylabel('Phase [°]')

        if errors:
            axes[0].annotate(self.timing_gain_fit.gain_summary(), (0.025, 0.95),
                             xycoords='axes fraction', ha='left', va='top')
            axes[1].annotate(self.timing_gain_fit.timing_summary(), (0.025, 0.95),
                             xycoords='axes fraction', ha='left', va='top')
        if errors:
            option_list.append(errors + '_errors')

        axes[0].set_xscale(scale)
        if scale != 'log':
            option_list.append(scale)

        axes[0].legend(loc='lower right')
        subplots_squeeze(fig, hspace=0)

        self._save_image(fig, option_list)


LOG_SETTINGS = {
    'version': 1,  # schema
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': LOG_FILE_NAME,
            'level': 'DEBUG',
            'formatter': 'detailed',
        },
    },
    'formatters': {
        'simple': {
            'format':
            '%(levelname)-8s %(funcName)s - %(message)s'
        },
        'detailed': {
            'format': '%(levelname)-8s %(filename)s:%(name)s:%(funcName)s - '
                      '%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'loggers': {
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
}


def main(argv=None):
    """Run analysis and return system exit code."""
    if argv is None:
        argv = sys.argv
    parser = _argparser()
    args = parser.parse_args(argv[1:])

    if os.path.isfile(LOG_FILE_NAME):
        os.remove(LOG_FILE_NAME)
    logging.config.dictConfig(LOG_SETTINGS)

    config = vars(args).copy()
    result = calibration_analyzer(**config)

    return len(result) == 0


if __name__ == '__main__':
    sys.exit(main())
