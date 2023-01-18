# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:00:55 2020

n.b. parent directory must be added to PYTHONPATH

@author: nackerle
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.markers import MarkerStyle
from scipy import stats

import seaborn as sns
import pandas as pd

sys.path.append("..")
from shared import get_logger, _annotate
from manufacturer.manufacturer_data import MANUFACTURER_PLUS_FILE

# %% constants
NEW_FILE = 'YKA_2015-2019.csv'
OLD_FILE = 'YKA_2015-2019_findpeaks.csv'
REFERNCE_DEGC = 0
CORRECTION_STR = 'corrected to %g°C' % REFERNCE_DEGC
KAPPA_MEASURED = -0.00223

COLORS = {'B': sns.color_palette()[0],
          'R': sns.color_palette()[3]}
STATS_FMT = '$\\mu$ = %.2f%%\n$\\sigma$ = %.2f%%'
OUTLIERS = ['YKAR3']

BASE_NAME = os.path.basename(os.path.splitext(__file__)[0])

# %% setup
logger = get_logger(__name__, BASE_NAME + '.log')
pd.plotting.register_matplotlib_converters()
logger.info('Loading: ' + MANUFACTURER_PLUS_FILE)
mfg_df = pd.read_csv(MANUFACTURER_PLUS_FILE, index_col=0)
nominal = mfg_df.loc['nominal']


# %% functions
def _unique(items):
    return ','.join(items.unique())


# %% load calibration results
cal_df = pd.read_csv(NEW_FILE, parse_dates=['start'])
cal_df['test name'] = (
    cal_df['test name'].str.split('\\', expand=True)[2].str.replace('_', ''))
cal_df.rename(columns={'test name': 'station_code'}, inplace=True)
cal_df['gain'] = 10**(cal_df['gain [dB]']/20)
cal_df['delta_t'] = cal_df['temperature [°C]'] - REFERNCE_DEGC
cal_df['year'] = cal_df.start.dt.year
cal_df.set_index(['year', 'station_code'], inplace=True, verify_integrity=True)
cal_df.sort_index(inplace=True)

# %% plot phase only
good_phase = cal_df['phase [°]'].between(-75, -70)
cal_df['phase offset [°]'] = (
    cal_df.loc[good_phase, 'phase [°]'] -
    cal_df.loc[good_phase, 'phase [°]'].mean())
cal_df['phase offset [°]'].hist()
ax = cal_df.plot.scatter(x='start', y='phase offset [°]')
ax.get_figure().savefig('phase_offset_vs_time.png', bbox_inches='tight')

# %% load old results
old_df = pd.read_csv(OLD_FILE, header=[0, 1], index_col=0)
old_df['CALIB'].columns = old_df['CALIB'].columns.astype(int)
check_df = (mfg_df[['NCALIB']].drop('nominal') /
            old_df['NCALIB'].rename(columns={'nominal': 'NCALIB'}))
assert (check_df - 0.990175).abs().max().squeeze() < 0.003
old_df = old_df['CALIB']
old_df.columns = old_df.columns.astype(int)
old_series = old_df.unstack()
old_series.name = 'findpeaks_calib'
cal_df = cal_df.join(old_series)
cal_df['start'] = cal_df['start'].dt.round('s')

# %% combine
df = pd.merge(cal_df, mfg_df, left_on='station_code', right_index=True)

df['findpeaks_room'] = df.findpeaks_calib*(1 + KAPPA_MEASURED*df.delta_t)
df['gain_room'] = df.gain*(1 - KAPPA_MEASURED*df.delta_t)

df['findpeaks_deviation'] = -100*(df.findpeaks_calib/df.NCALIB - 1)
df['findpeaks_room_deviation'] = -100*(df.findpeaks_room/df.NCALIB - 1)
df['gain_deviation'] = 100*(df.gain/df.gain_expected - 1)
df['gain_room_deviation'] = 100*(df.gain_room/df.gain_expected - 1)

aggregators = {
    column: _unique if type_ == object else
    ('min' if column == 'start' else 'mean')
    for column, type_ in df.dtypes.items()}
df_mean = df.groupby('station_code').agg(aggregators)
df_mean.index = pd.MultiIndex.from_product((['mean'], df_mean.index))
df = pd.concat((df, df_mean)).sort_index()

ref = 2015
df['findpeaks_drift'] = df.findpeaks_deviation - df.findpeaks_deviation[ref]
df['findpeaks_room_drift'] = (df.findpeaks_room_deviation -
                              df.findpeaks_room_deviation[ref])
df['gain_drift'] = df.gain_deviation - df.gain_deviation[ref]
df['gain_room_drift'] = df.gain_room_deviation - df.gain_room_deviation[ref]

output_csv = BASE_NAME + '.csv'
logger.info('Saving: ' + output_csv)
df.to_csv(output_csv)


# %% make table for publication
summary_df = df[df.index.get_level_values('year')
                != 'mean'].reset_index().pivot(
    index='station_code', columns='year',
    values=['gain', 'gain_deviation', 'gain_room_drift'])
summary_df.gain = summary_df.gain.round(3)
summary_df.gain_deviation = summary_df.gain_deviation.round(1)
summary_df.gain_room_drift = summary_df.gain_room_drift.round(1)
summary_df.drop(
    columns=[('gain', item) for item in summary_df.gain.columns
             if item != summary_df.gain.columns[0]],
    inplace=True)
summary_df.drop(
    columns=[('gain_deviation', summary_df.gain_deviation.columns[0]),
             ('gain_room_drift', summary_df.gain_room_drift.columns[0])],
    inplace=True)
summary_df.rename(columns={'gain_deviation': 'rel. [%]',
                           'gain_room_drift': 'drift [%]'}, inplace=True)
summary_df.columns = [' '.join([str(item) for item in items])
                      for items in list(summary_df.columns)]

summary_csv = BASE_NAME + '_summary.csv'
logger.info('Saving: ' + summary_csv)
summary_df.reset_index().to_csv(summary_csv, index=False)

# %% plot times
time_df = df[df.index.get_level_values('year') != 'mean'].reset_index().pivot(
    index='station_code', columns='year', values='start')

line_number = {'B': 0, 'R': 10}
station_number = {item: int(item[-1]) + line_number[item[-2]]
                  for item in time_df.index}

fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))
t_zero = pd.to_datetime('1970-01-01')
for (year, series), marker in zip(time_df.iteritems(),
                                  MarkerStyle.filled_markers):
    series = series.sort_values().copy()
    ax.plot(t_zero + (series - pd.to_datetime(series.dt.date.min())),
            series.index.map(station_number),
            label=year, marker=marker, markerfacecolor='white')
ax.set_xlabel('Time of day')
ax.xaxis_date()
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.set_xlim((t_zero + pd.to_timedelta(15, 'hours'),
             t_zero + pd.to_timedelta(23.999, 'hours')))
ticks = np.arange(series.index.map(station_number).min(),
                  series.index.map(station_number).max() + 1)
labels = [('YKA%s%d' % (('B' if item < 10 else 'R'), item % 10))
          for item in ticks]
labels = [item if item in series.index else '' for item in labels]
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.legend(title='Year')

output_png = 'calibration_time.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')

# %% plot baseline calibrations
fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.5), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05)

SUBSET_NOTES = (
        ('findpeaks_deviation', 'time-domain'),
        ('findpeaks_room_deviation', 'time-domain\n' + CORRECTION_STR),
        ('gain_room_deviation', 'frequency-domain\n' + CORRECTION_STR))

for (index, note), ax in zip(SUBSET_NOTES, axes):
    sns.distplot(df[index][ref], ax=ax, color='0.5',
                 vertical=False, hist=True, rug=True, kde=False,
                 fit=stats.norm, fit_kws=dict(linestyle='--'))
    sns.distplot(df[index][ref].drop(OUTLIERS), ax=ax, color='0.5',
                 vertical=False, hist=False, rug=False, kde=False,
                 fit=stats.norm)

    _annotate(ax, STATS_FMT % stats.norm.fit(df[index][ref].drop(OUTLIERS)),
              'upper right')
    _annotate(ax, note, 'upper left')

    ax.set_xlabel('Gain Deviation from Nominal (%)')
    ax.set_yticks([])
    ax.set_xlim((-15, 15))

sns.despine(fig)

output_png = 'baseline_variation.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')

# %% plot temporal variation
fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5),
                         sharex='col', sharey='row',
                         gridspec_kw=dict(width_ratios=[3, 1]))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

SUBSET_NOTES = (
        ('findpeaks_drift', 'time-domain'),
        ('findpeaks_room_drift', 'time-domain\n' + CORRECTION_STR),
        ('gain_room_drift', 'frequency-domain\n' + CORRECTION_STR))

for station, stn_df in df.drop('mean').groupby('station_code'):
    color = COLORS[station[-2]]
    marker = MarkerStyle.filled_markers[int(station[-1])]
    face_color = 'none' if station[-2] == 'R' else color
    for (index, note), ax in zip(SUBSET_NOTES, axes[:, 0]):
        _annotate(ax, note, 'upper left')
        ax.plot(stn_df.start, stn_df[index], label=station,
                ls='none', marker=marker, c=color, mfc=face_color)

for ax in axes[:, 0]:
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))

for (index, _), ax in zip(SUBSET_NOTES, axes[:, 1]):
    sns.distplot(df[index].drop(ref), ax=ax, vertical=True,
                 rug=True, kde=False, fit=stats.norm, color='0.5')
    _annotate(ax, STATS_FMT % stats.norm.fit(
        df[index].drop(ref)), 'upper right')

    ax.set_ylabel('')
    ax.set_xticks([])

axes[1, 0].set_ylabel('Gain deviation from %s [%%]' % ref)

lines, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 1].legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 1))
sns.despine(fig)

output_png = 'temporal_variation.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')

# %% plot gain vs. generator constant deviation
fig, ax = plt.subplots()
for year, year_df in df.groupby(df.start.dt.year):
    ax.scatter(100*(year_df.Kg/nominal.Kg - 1),
               100*(year_df.gain/nominal.gain_expected - 1),
               label=year)
ax.plot([-5, 3], [-10, 6], '--k', label='2:1')
ax.set_xlabel('Generator constant deviation [%]')
ax.set_ylabel('Gain deviation [%]')
ax.legend()

year_df = df[df.start.dt.year == 2017]
sns.despine(fig)

output_png = 'generator_variation.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')
