# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:24:37 2020

@author: nackerle
"""
# mypy: ignore-errors
import numpy as np
from matplotlib.offsetbox import AnchoredText

from calan.utilities import stdval  # type: ignore

# %% constants
ROOM_TEMPERATURE_DEGC = 20

STATS_FMT = '$\\mu$ = %.2f%%\n$\\sigma$ = %.2f%%'


def _annotate(ax, text, loc='upper center'):
    ax.add_artist(AnchoredText(text, loc=loc, frameon=False))


def _nice_bins(measurements, target=30):
    range_ = np.max(measurements) - np.min(measurements)
    bin_size = stdval(range_/target, 3)
    min_val = bin_size*np.floor(np.min(measurements)/bin_size)
    max_val = bin_size*np.ceil(np.max(measurements)/bin_size)
    return np.arange(min_val, max_val + bin_size/2, bin_size)


def inventory_items(inventory):
    '''
    Iterate through contents of an inventory
    '''
    for network in inventory:
        for station in network:
            for channel in station:
                yield network, station, channel
