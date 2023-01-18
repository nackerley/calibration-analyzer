# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:24:37 2020

@author: nackerle
"""
import os
import logging
from logging.config import dictConfig
import numpy as np
from matplotlib.offsetbox import AnchoredText

from catalogue_tools.utilities import stdval

# %% constants
TCR_CU_OHM_DEGC = 0.00393
ALPHA_BR_ALNICO = -0.0002
ROOM_TEMPERATURE_DEGC = 20

STATS_FMT = '$\\mu$ = %.2f%%\n$\\sigma$ = %.2f%%'

LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
LOG_SETTINGS = {
    'version': 1,  # logging schema
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'mode': 'w',
            },
        },
    'formatters': {
        'simple': {
            'format':
            '%(levelname)-8s %(filename)s:%(name)s:%(funcName)s - %(message)s'
            },
        },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
            },
        }
    }


# %% definitions
def _annotate(ax, text, loc='upper center'):
    ax.add_artist(AnchoredText(text, loc=loc, frameon=False))


def _nice_bins(measurements, target=30):
    range_ = np.max(measurements) - np.min(measurements)
    bin_size = stdval(range_/target, 3)
    min_val = bin_size*np.floor(np.min(measurements)/bin_size)
    max_val = bin_size*np.ceil(np.max(measurements)/bin_size)
    return np.arange(min_val, max_val + bin_size/2, bin_size)


def get_logger(name, log_file_name=None, log_console_level='INFO'):
    '''
    Return named logger which logs to console and file.

    Configures root logger for log_file_name and log_console_level.
    Logging to file is always at DEBUG level.
    '''
    assert log_console_level in LOG_LEVELS

    handlers = logging.getLogger().handlers
    not_previously_configured = len(handlers) == 0

    if not_previously_configured and not log_file_name:
        raise ValueError(
            'Must specify log filename if logging not previouesly configured.')

    if not_previously_configured:
        LOG_SETTINGS['handlers']['console'].update(
            {'level': log_console_level})
        LOG_SETTINGS['handlers']['file'].update(
            {'filename': log_file_name})
        dictConfig(LOG_SETTINGS)

    logger = logging.getLogger(name)
    if not_previously_configured:
        logger.info('Logfile: ' + os.path.abspath(log_file_name))
    return logger


def inventory_items(inventory):
    '''
    Iterate through contents of an inventory
    '''
    for network in inventory:
        for station in network:
            for channel in station:
                yield network, station, channel


