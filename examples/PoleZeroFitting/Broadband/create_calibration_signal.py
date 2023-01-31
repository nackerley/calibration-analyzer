#!/usr/bin/env python3

import argparse
import math
import os
import struct
import sys

MAX_SEISMOMETER_AMPLITUDE = -5.0  # negative polarity
MAX_ACCELEROMETER_AMPLITUDE = 2.0


class Generator(object):
    SAMPLE_RATE = 30000
    MAX_SAMPLE = 2**15 - 1

    def __init__(self, duration, amplitude, max_amplitude, output_file, meta):
        self.duration = duration
        self.amplitude = amplitude
        self.max_amplitude = max_amplitude
        self.output_file = output_file
        self.meta = meta

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.output_file.close()

    def write_sample(self, sample):
        raw_sample = int(round((sample * (self.amplitude / self.max_amplitude) * self.MAX_SAMPLE)))
        raw_sample = struct.pack("<h", raw_sample)
        self.output_file.write(raw_sample)

    def write_meta(self):
        if self.meta:
            dirname = os.path.dirname(self.output_file.name)
            filename = os.path.basename(self.output_file.name)
            durationMillis = self.duration * 1000
            meta_file = open(dirname + ".meta_" + filename, "w", encoding='UTF-8')
            meta_file.write("file=%s\n" % filename)
            meta_file.write("durationMillis=%d\n" % durationMillis)
            meta_file.close()

    def numSamples(self):
        return Generator.SAMPLE_RATE * self.duration


class SineGenerator(Generator):
    def __init__(self, frequency, duration, amplitude, max_amplitude, file_name, meta):
        super(SineGenerator, self).__init__(duration, amplitude, max_amplitude, file_name, meta)
        if frequency is None:
            self.frequency = 1
        else:
            self.frequency = frequency

    def create(self):
        for i in range(self.numSamples()):
            self.write_sample(math.sin(float(i) / Generator.SAMPLE_RATE * self.frequency * 2 * math.pi))
        self.write_meta()


class StepGenerator(Generator):
    def __init__(self, duration, amplitude, max_amplitude, file_name, meta):
        super(StepGenerator, self).__init__(duration, amplitude, max_amplitude, file_name, meta)

    def create(self):
        num_samples = self.numSamples()
        for _ in range(int(num_samples / 2)):
            self.write_sample(0)
        for _ in range(int(num_samples / 2), num_samples):
            self.write_sample(1)
        self.write_meta()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate a sensor calibration file to play from a '
        'Nanometrics digitizer')
    required_args = parser.add_argument_group('Required arguments')
    required_args.add_argument(
        'sensor', choices=['seismometer', 'accelerometer'],
        help='specify sensor type')
    required_args.add_argument(
        'type', choices=['sine', 'step'], help='specify signal type')
    required_args.add_argument(
        'duration', type=int, help='duration in seconds')
    required_args.add_argument(
        'amplitude', type=float,
        help='amplitude in Volts (if seismometer max 5) or G '
        '(if accelerometer max 2)')
    required_args.add_argument(
        'output_file', type=argparse.FileType('wb'),
        help='name of output file')
    parser.add_argument(
        '--frequency', type=int, help='sine wave frequency (default 1 Hz)')
    parser.add_argument(
        '--meta', action='store_true', help='produce .meta file')
    args = parser.parse_args()

    max_amplitude = (MAX_SEISMOMETER_AMPLITUDE if args.sensor == 'seismometer'
                     else MAX_ACCELEROMETER_AMPLITUDE)

    if args.type == 'sine':
        with SineGenerator(args.frequency, args.duration, args.amplitude,
                           max_amplitude, args.output_file, args.meta) as generator:
            generator.create()
    elif args.type == 'step':
        if args.frequency is not None:
            print('Frequency argument is not compatible with a step function')
            sys.exit(1)
        with StepGenerator(args.duration, args.amplitude, max_amplitude,
                           args.output_file, args.meta) as generator:
            generator.create()
