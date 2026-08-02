# -*- coding: utf-8 -*-
"""
Generate a histogram demonstrating repeatability of calibration results.

n.b. parent directory must be added to PYTHONPATH

Created on Mon Nov 25 16:47:03 2019

@author: nackerle
"""
import os
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from shared import get_logger, ROOM_TEMPERATURE_DEGC, STATS_FMT, _annotate
from manufacturer.manufacturer_data import MANUFACTURER_PLUS_FILE


# %% setup
logger = get_logger(__name__,
                    os.path.basename(os.path.splitext(__file__)[0]) + '.log')
pd.plotting.register_matplotlib_converters()

# %% constants
OLD_FILE = 'outdoor_vault_temperature.csv'
NEW_FILE = 'outdoor_case_temperature.csv'
PROBE_POSITIONS = ['air', 'vault', 'sensor']
PROBE_COLORS = {position: color for position, color in
                zip(PROBE_POSITIONS, sns.color_palette())}
TEST_ID = 'TEST1'

# %% setup
logger.info('Loading: ' + MANUFACTURER_PLUS_FILE)
mfg_df = pd.read_csv(MANUFACTURER_PLUS_FILE, index_col=0)
nominal = mfg_df.loc['nominal']

logger.info('Nominal calibration temperature coefficient: %.3g%%/°C' %
            (100*nominal.kappa_c))
logger.info('Nominal ground motion temperature coefficient: %.3g%%/°C' %
            (100*nominal.kappa_g))

# %% combine old and new results
df_old = pd.read_csv(OLD_FILE, parse_dates=['start'], index_col='start')
df_old.rename(columns={'temperature [°C]': 'air temperature [°C]'},
              inplace=True)
df_old['probe temperature [°C]'] = df_old['test name'].str.split(
    '_', expand=True)[1].astype(float)
df_old['position'] = 'vault'


df_new = pd.read_csv(NEW_FILE, parse_dates=['start'], index_col='start')
df_new.rename(columns={'temperature [°C]': 'air temperature [°C]'},
              inplace=True)
df_new['probe temperature [°C]'] = df_new['test name'].str.split(
    '_', expand=True)[1].astype(float)
df_new['position'] = 'sensor'

df = pd.concat([df_old, df_new], sort=False)
df = df[df['f_peak [Hz]'] == 4]
df.sort_index(inplace=True)

# %% load manufacturer's calibration data
expected = mfg_df.loc[TEST_ID]
logger.info(TEST_ID + '\n' + str(expected))
logger.info('Expected %s ground motion temperature coefficient: %.3g%%/°C' %
            (TEST_ID, 100*expected.kappa_c))

# %% post-processing
df['gain'] = 10**(df['gain [dB]']/20)
df['gain deviation from nominal [%]'] = 100*(
    df['gain']/nominal.gain_expected - 1)
df['gain deviation from expected [%]'] = 100*(
    df['gain']/expected.gain_expected - 1)

df_air = df.drop(columns='probe temperature [°C]').rename(
    columns={'air temperature [°C]': 'temperature [°C]'})
df_air['position'] = 'air'
df_probe = df.drop(columns='air temperature [°C]').rename(
    columns={'probe temperature [°C]': 'temperature [°C]'})
df_long = pd.concat([df_air, df_probe], sort=False)

# %% compare temperatures
fig, ax = plt.subplots(figsize=(4.5, 4.5))

for position, df_probe in df.groupby('position'):
    ax.scatter(df_probe['air temperature [°C]'],
               df_probe['probe temperature [°C]'],
               label=position)
ax.set_xlabel('Yellowknife airport temperature [°C]')
ax.set_ylabel('probe temperature [°C]')
ax.legend()

sns.despine(fig)
output_png = 'temperatures.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=150, bbox_inches='tight')

# %% plot gain vs. temperature
fig, ax = plt.subplots(figsize=(4.5, 4.5))

for position in PROBE_POSITIONS:
    df_probe = df_long[df_long['position'] == position]
    ax.scatter(df_probe['temperature [°C]'],
               df_probe['gain deviation from expected [%]'],
               label=position)
ax.set_xlabel('temperature [°C]')
ax.set_ylabel('gain deviation from expected [%]')
ax.legend()

sns.despine(fig)

output_png = 'temp_vs_gain.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=150, bbox_inches='tight')

# %% fit gain vs. temperature
fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.set_ylim((0, 10))
ax.set_xlim((-40, 10))

fits = []
for position in PROBE_POSITIONS:
    df_probe = df_long[df_long['position'] == position]
    fit = sm.OLS(df_probe['gain deviation from expected [%]'],
                 sm.add_constant(df_probe['temperature [°C]'])).fit()
    label = '%s: $R^2$=%.2g\n$\\mu$=%.3f%%/°C' % (
        position, fit.rsquared, fit.params[1])

    sns.regplot(x='temperature [°C]', y='gain deviation from expected [%]',
                data=df_probe, truncate=False,
                label=label, ax=ax, color=PROBE_COLORS[position])
    fits.append(fit)

ax.legend(loc='upper right', title='probe position')
sns.despine(fig)

output_png = 'temp_vs_gain_fitted.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=150, bbox_inches='tight')


# %% gaussian fits to temperature corrected gain
for position, fit in zip(PROBE_POSITIONS, fits):
    KAPPA_MEASURED = fit.params[1]/100
    GAIN_DEVIATION_ROOM = fit.params[1]*ROOM_TEMPERATURE_DEGC + fit.params[0]
    MFG_ROOM_TEMPERATURE_DEGC = -fit.params[0]/fit.params[1]
    logger.info('Probe position: ' + position)
    logger.info('Measured calibration temperature coefficient: %.3g%%/°C' %
                (100*KAPPA_MEASURED))
    logger.info("Gain deviation at %.3g°C: %.3g%%" %
                (ROOM_TEMPERATURE_DEGC, GAIN_DEVIATION_ROOM))
    logger.info("Apparent manufacturer's calibration temperature: %.1f°C" %
                (MFG_ROOM_TEMPERATURE_DEGC))

df_long['delta_t'] = df_long['temperature [°C]'] - MFG_ROOM_TEMPERATURE_DEGC
df_long['gain_room'] = df_long.gain*(1 - KAPPA_MEASURED*df_long.delta_t)
df_long['room temperature gain deviation from nominal [%]'] = 100*(
    df_long.gain_room/nominal.gain_expected - 1)
df_long['room temperature gain deviation from expected [%]'] = 100*(
    df_long.gain_room/expected.gain_expected - 1)

param = 'room temperature gain deviation from expected [%]'
# gain_bins = _nice_bins(df_long[param], target=15)

fig, axes = plt.subplots(len(PROBE_POSITIONS), 1, sharex=True)
fig.subplots_adjust(hspace=0.15)

for i, (position, ax) in enumerate(zip(PROBE_POSITIONS, axes)):
    df_probe = df_long[df_long['position'] == position]

    sns.distplot(df_probe[param], bins='auto', ax=ax, rug=True, kde=False,
                 fit=stats.norm, color=PROBE_COLORS[position])

    ax.set_ylabel(position + ' temp.')
    ax.set_yticklabels([])
    if i < len(PROBE_POSITIONS) - 1:
        ax.set_xlabel('')

    _annotate(ax, STATS_FMT % stats.norm.fit(df_probe[param]), 'upper right')

sns.despine(fig)
output_png = 'outdoor_gaussian_fits.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=150, bbox_inches='tight')

# %% assess temperature error
df_long['inferred temperature [°C]'] = (
    df_long['gain deviation from expected [%]'] - fit.params[0])/fit.params[1]
TEMP_FMT = '$\\mu$ = %.2f°C\n$\\sigma$ = %.2f°C'

fig, axes = plt.subplots(len(PROBE_POSITIONS), 1, sharex=True)
fig.subplots_adjust(hspace=0.15)

for i, (position, ax) in enumerate(zip(PROBE_POSITIONS, axes)):
    df_probe = df_long[df_long['position'] == position]
    data = df_probe['temperature [°C]'] - df_probe['inferred temperature [°C]']

    sns.distplot(data, bins='auto', ax=ax, rug=True, kde=False, fit=stats.norm,
                 color=PROBE_COLORS[position])

    ax.set_ylabel(position + ' temp.')
    ax.set_yticklabels([])
    if i < len(PROBE_POSITIONS) - 1:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('inferred temperature error [°C]')

    _annotate(ax, TEMP_FMT % stats.norm.fit(data), 'upper right')

sns.despine(fig)
output_png = 'outdoor_temperatures.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=150, bbox_inches='tight')
