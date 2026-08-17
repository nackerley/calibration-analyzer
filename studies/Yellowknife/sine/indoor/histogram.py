# -*- coding: utf-8 -*-
"""
Generate a histogram demonstrating repeatability of calibration results.

Created on Mon Nov 25 16:47:03 2019

@author: nackerle
"""
# mypy: ignore-errors
import os
from scipy import stats
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from calan.core import start_logger
from ..shared import _nice_bins, _annotate, STATS_FMT
from ..manufacturer.manufacturer_data import MANUFACTURER_PLUS_FILE

# %% constants
NEW_FILE = 'sine_analyzer_Indoor.csv'
OLD_FILE = 'indoor-outdoor_ncalib.csv'

# %% setup
logger = start_logger(
    __name__, os.path.basename(os.path.splitext(__file__)[0]) + '.log', 'INFO')
logger.info('Loading: %s', MANUFACTURER_PLUS_FILE)
mfg_df = pd.read_csv(MANUFACTURER_PLUS_FILE, index_col=0)
nominal = mfg_df.loc['nominal']

# %% combine old and new results
df_gain = pd.read_csv(NEW_FILE, parse_dates=['start'])
df_gain['test name'] = df_gain['test name'].str.split(
    '_', expand=True)[1].astype(int)
df_gain.set_index('test name', inplace=True)
df_gain.sort_index(inplace=True)

df_ncalib = pd.read_csv(OLD_FILE)
df_ncalib['test name'] = df_ncalib['test name'].str.split(
    '_', expand=True)[1].astype(int)
df_ncalib.set_index('test name', inplace=True)
df_ncalib.drop(columns=['start'], inplace=True)
df_ncalib.sort_index(inplace=True)

df = df_gain.join(df_ncalib)
df.drop([31, 33, 34], inplace=True)

# %%
df['gain'] = 10**(df['gain [dB]']/20)

df['ncalib_deviation'] = -100*(1 - df['ncalib']/nominal.NCALIB)
df['gain_deviation'] = 100*(1 - df['gain']/nominal.gain_expected)

logger.info(df[['ncalib_deviation', 'gain_deviation']]
            .describe(percentiles=[]).to_string(float_format='%.2g'))

# %% binning
ncalib_bins = _nice_bins(df['ncalib_deviation'], target=15)
gain_bins = _nice_bins(df['gain_deviation'], target=15)

# %% gaussian fits
fig, axes = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.15)
sns.distplot(df['ncalib_deviation'], bins=ncalib_bins, ax=axes[0],
             rug=True, kde=False, fit=stats.norm, color='0.5')
sns.distplot(df['gain_deviation'], bins=gain_bins, ax=axes[1],
             rug=True, kde=False, fit=stats.norm, color='0.5')

axes[0].set_ylabel('old method')
axes[0].set_xlabel('')
axes[0].set_yticklabels([])
_annotate(axes[0], STATS_FMT % stats.norm.fit(df['ncalib_deviation']),
          'upper right')
axes[1].set_ylabel('new method')
axes[1].set_xlabel('deviation from nominal [%]')
axes[1].set_yticklabels([])
_annotate(axes[1], STATS_FMT % stats.norm.fit(df['gain_deviation']),
          'upper right')
sns.despine(fig)
output_png = 'indoor_gaussian_fits.png'
logger.info('Saving: %s', output_png)
fig.savefig(output_png, dpi=150, bbox_inches='tight')

# %% joint plots
g = sns.jointplot(x='gain_deviation', y='ncalib_deviation', data=df,
                  kind='kde')
g.plot_joint(plt.scatter, s=30, linewidth=1, marker='+')
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels('deviation [%] (new)', 'deviation [%] (old)')

output_png = 'joint_plot_1.png'
logger.info('Saving: %s', output_png)
g.fig.savefig(output_png, dpi=150, bbox_inches='tight')

g = sns.jointplot(x='gain_deviation', y='ncalib_deviation', data=df,
                  annot_kws=dict(stat="r"),
                  s=30, linewidth=1, marker='+')
g.set_axis_labels('deviation [%] (new)', 'deviation [%] (old)')

output_png = 'joint_plot_2.png'
logger.info('Saving: %s', output_png)
g.fig.savefig(output_png, dpi=150, bbox_inches='tight')
