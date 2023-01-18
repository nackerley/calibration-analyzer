# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:17:06 2020

@author: nackerle
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from obspy import read_inventory, UTCDateTime

from shared import get_logger
from manufacturer_data import MANUFACTURER_PLUS_FILE, OLD_XML, NEW_XML, NCALPER

# %% constants
BASE_NAME = os.path.basename(os.path.splitext(__file__)[0])
GAIN_LIMS = [-2, 4]
FREQ_LIMS = np.array([0.7, 20])
FREQUENCIES = np.logspace(np.log10(0.7), np.log10(20), 1000)


# %% definitions
def plot_inventory(inventory):
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    for seed_id in inventory.get_contents()['channels']:
        response = inventory.get_response(seed_id, UTCDateTime('2020-01-01'))
        nominal_gain = response.instrument_sensitivity.value
        tf = response.get_evalresp_response_for_frequencies(
            FREQUENCIES)
        axes[0].semilogx(FREQUENCIES, 100*(np.abs(tf)/nominal_gain - 1),
                         label=seed_id)
        axes[1].semilogx(FREQUENCIES, np.angle(tf, deg=True), label=seed_id)

    f_norm = response.response_stages[0].stage_gain_frequency
    axes[0].axvline(f_norm, linestyle='--', color='black', linewidth=0.5,
                    label='sensor gain')
    axes[1].axvline(f_norm, linestyle='--', color='black', linewidth=0.5,
                    label='sensor gain')
    axes[0].axvline(1/NCALPER, linestyle=':', color='black', linewidth=0.5,
                    label='calibration')
    axes[1].axvline(1/NCALPER, linestyle=':', color='black', linewidth=0.5,
                    label='calibration')

    axes[0].set_ylim(GAIN_LIMS)
    axes[0].set_ylabel('Magnitude Deviation [%]')
    axes[1].set_ylabel('Phase [°]')
    axes[1].set_ylim([-181, 181])
    axes[1].set_yticks(np.arange(-180, 181, 45))
    axes[1].set_xlim((FREQUENCIES.min(), FREQUENCIES.max()))
    axes[1].set_xlabel('Frequency [Hz]')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    return fig


# %% setup
logger = get_logger(__name__, BASE_NAME + '.log')
logger.info('Loading: ' + MANUFACTURER_PLUS_FILE)
mfg_df = pd.read_csv(MANUFACTURER_PLUS_FILE, index_col=0)

# %% summarize calibration data
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)
fig.subplots_adjust(wspace=0.01)

mfg_cal_df = mfg_df.loc[mfg_df.index != 'nominal']
axes[0].plot(mfg_cal_df.Rg, mfg_cal_df.Kg, 'ok',
             markerfacecolor='none')
axes[0].set_xlabel(r'Generator resistance, Rg [$\Omega$]')
axes[0].set_ylabel('Generator constant, Kg [V/(m/s)]')
axes[0].axvline(mfg_df.at['nominal', 'Rg'],
                ls='--', color='0.5', label='nominal')
axes[0].axhline(mfg_df.at['nominal', 'Kg'],
                ls='--', color='0.5')
axes[0].legend(loc='upper center')

axes[1].plot(mfg_cal_df.CDR, mfg_cal_df.Kg, 'xk')
axes[1].set_xlabel(r'Critical damping resistance, CDR [$\Omega$]')
axes[1].axvline(mfg_df.at['nominal', 'CDR'],
                ls='--', color='0.5', label='nominal')
axes[1].axhline(mfg_df.at['nominal', 'Kg'],
                ls='--', color='0.5')

sns.despine(fig)
output_png = BASE_NAME + '.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')

# %% plot responses
old_inventory = read_inventory(OLD_XML)
new_inventory = read_inventory(NEW_XML)
fig = plot_inventory(old_inventory)
output_png = 'response_existing.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')

fig = plot_inventory(new_inventory)
output_png = 'response_new.png'
logger.info('Saving: ' + output_png)
fig.savefig(output_png, dpi=300, bbox_inches='tight')
