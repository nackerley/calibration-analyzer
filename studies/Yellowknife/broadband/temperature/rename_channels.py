from glob import glob
from obspy import read

RENAME = {
    '.3082..HHZ': 'XX.YKAT1..SHZ',
    '.3082..HHC': 'XX.YKAT1..SHC',
}

for waveform_file in glob('*.mseed'):
    print(f'Reading: {waveform_file}')
    stream = read(waveform_file)
    for trace in stream:
        try:
            print(f'Renaming: {trace.id} as {RENAME[trace.id]}')
            trace.id = RENAME[trace.id]
        except KeyError:
            print(f'Unknown, skpping: {trace.id}')
    print(f'Writing: {waveform_file}')
    stream.write(waveform_file)
