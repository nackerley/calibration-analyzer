"""Read raw calibration results and make summary plot."""
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=['#777777', '#444444',  '#111111'],
    linestyle=['-.', '--', '-'],
)
plt.rcParams['lines.linewidth'] = 1.0

INPUT_CSV = 'arbitrary_analyzer.csv'
data = pd.read_csv(INPUT_CSV, index_col=[0, 1])

# extract temperature from filenames
temps = (
    data.loc[('info', 'waveform_file'), :].fillna('_NaN_')
    .str.split('_', n=2, expand=True)[1].apply(float))
data.columns = pd.Index(
    ['nominal' if np.isnan(temp) else f'{temp:.1f}°C' for temp in temps])

# subset to min, max, room and nominal
keep = list(data.columns[temps.isin([
    temps.min(),
    temps.max(),
    # temps[(temps - 15).abs().idxmin()],
    np.nan])])
data = data.loc[:, keep]

magnitude_db = data.loc['magnitude_db'].astype(float)
magnitude_db.index = pd.Index(magnitude_db.index.astype(float), name='f_hz')
phase_deg = data.loc['phase_deg'].astype(float)
phase_deg.index = pd.Index(phase_deg.index.astype(float), name='f_hz')
variance_db = data.loc['variance_db'].astype(float)
variance_db.index = pd.Index(variance_db.index.astype(float), name='f_hz')

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.5, 6.5))
fig.subplots_adjust(hspace=0)
mag_ax, phase_ax, var_ax = axes

magnitude_db.plot(ax=mag_ax, logx=True)
mag_ax.set_ylabel(r'Mag. (dB wrt V·s/m)')
mag_ax.set_yticks(np.arange(0, 60, 5))
mag_ax.set_ylim(30.1, 54.9)
phase_deg.plot(ax=phase_ax, logx=True)
phase_ax.set_ylabel('Phase (°)')
phase_ax.set_yticks(np.arange(0, 181, 45))
phase_ax.set_ylim(-10, 190)
phase_ax.get_legend().remove()
variance_db.plot(ax=var_ax, logx=True)
var_ax.set_ylabel('Variance (dB)')
var_ax.set_xlabel('Frequency (Hz)')
var_ax.set_xlim(variance_db.index.min(), variance_db.index.max())
var_ax.get_legend().remove()
var_ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

output_png = Path(__file__).with_suffix('.png')
print(f'Saving: {output_png}')
fig.savefig(output_png, bbox_inches='tight')
