# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:19:39 2020

@author: nackerle
"""
# mypy: ignore-errors
import os
from typing import Iterator
from json import dumps
from getpass import getuser
from copy import deepcopy

import numpy as np
from scipy.signal import ZerosPolesGain
import pandas as pd

from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import (
    Channel, Inventory, Network, Response, PolesZerosResponseStage,
    InstrumentSensitivity, Comment, Person, Station)
from obspy.clients.fdsn import Client

from calan.utilities import round_sig  # type: ignore
from calan.core import start_logger  # type: ignore

# constants
current_dir = os.path.dirname(os.path.abspath(__file__))
MANUFACTURER_FILE = os.path.join(current_dir, 'CalibrationSheets.csv')
MANUFACTURER_PLUS_FILE = os.path.splitext(MANUFACTURER_FILE)[0] + 'Derived.csv'
CHIS_PUBLIC_FDSNWS = 'https://www.earthquakescanada.nrcan.gc.ca'

TCR_CU_OHM_DEGC = 0.00393
ALPHA_BR_ALNICO = -0.0002

NETWORK = 'CN'
STATION_PREFIX = 'YKA'
PRECISION = 5
START = UTCDateTime('2015-11-23 16:28:19')
END = UTCDateTime('2019-03-06 21:40:26')
NCALPER = 0.25

NRL_URL = (
    'http://service.iris.edu/irisws/nrl/1/combine'
    '?instconfig={instconfig}&format=stationxml')
DATALOGGER_ID = (
    'datalogger_Guralp_CMG-DM24-Mk3-Variable_PG4_'
    'TL31_TP1000-200-40-20-10-5_FR40')

OLD_XML = f'{NETWORK}.{STATION_PREFIX}_existing.xml'
NEW_XML = f'{NETWORK}.{STATION_PREFIX}_new.xml'

BASE_NAME = os.path.join(current_dir,
                         os.path.basename(os.path.splitext(__file__)[0]))

np.seterr(all='raise')

if getuser() == 'nackerle':
    AUTHOR = Person(emails=['nicholas.ackerley@nrcan-rncan.gc.ca'])
else:
    raise RuntimeError(f'Add email for user: {getuser()}')


def inventory_items(
    inv: Inventory,
) -> Iterator[tuple[Network, Station, Channel]]:
    """Iterate through network, station, channel of an inventory."""
    for net in inv:
        for sta in net:
            for chan in sta:
                yield net, sta, chan


def stage_params(zeros, poles, gain, norm_freq, gain_freq):
    """
    Compute stage normalization factor and gain at normalization frequency.
    """
    zpk_unity = ZerosPolesGain(zeros, poles, 1)
    norm_factor = 1/np.abs(zpk_unity.freqresp(2*np.pi*norm_freq)[1][0])
    zpk_gain = ZerosPolesGain(zeros, poles, norm_factor)
    gain_at_gain_freq = gain/np.abs(zpk_gain.freqresp(2*np.pi*gain_freq)[1][0])
    return norm_factor, gain_at_gain_freq


if __name__ == '__main__':

    # FDSNWS for our responses
    logger = start_logger(__name__, BASE_NAME + '.log', 'INFO')
    logger.info('Client: %s', CHIS_PUBLIC_FDSNWS)
    client = Client(CHIS_PUBLIC_FDSNWS)

    # nominal response library
    nrl_datalogger_stages = read_inventory(NRL_URL.format(
        instconfig=DATALOGGER_ID))[0][0][0].response.response_stages

    # load data
    logger.info('Loading: %s', MANUFACTURER_FILE)
    mfg_df = pd.read_csv(MANUFACTURER_FILE, index_col='station_code')
    nominal = mfg_df.loc['YKNOM']
    for column in mfg_df.columns:
        if isinstance(nominal[column], float) and not pd.isnull(
                nominal[column]):
            mfg_df[column].fillna(nominal[column], inplace=True)

    # compute derived parameters
    mfg_df.loc['ideal'] = mfg_df.loc['YKNOM']
    mfg_df.at['ideal', 'Rd'] = (
        np.sqrt(2)*mfg_df.at['ideal', 'CDR'] -
        mfg_df.at['ideal', 'Rg']).round()

    mfg_df['Rc'] = (mfg_df.Rm + mfg_df.Rp)/mfg_df.Kc
    mfg_df['Sg'] = mfg_df.Kg/(1 + mfg_df.Rg/mfg_df.Rd)
    mfg_df['lambda0'] = mfg_df.CDR/(mfg_df.Rg + mfg_df.Rd)
    mfg_df['Sd'] = mfg_df.Kd/mfg_df.Si
    mfg_df['kappa_g'] = -TCR_CU_OHM_DEGC*(
        1/(1 + (mfg_df.Rd/mfg_df.Rg))) + ALPHA_BR_ALNICO
    mfg_df['kappa_c'] = mfg_df['kappa_g'] - TCR_CU_OHM_DEGC*(
        1/(1 + (mfg_df.Rp/mfg_df.Rm))) + ALPHA_BR_ALNICO

    mfg_df['NCALPER'] = NCALPER
    mfg_df['NCALIB0'] = 1e9/(2*np.pi/NCALPER*mfg_df.Sd*mfg_df.Kp*mfg_df.Sg)
    mfg_df['gain0'] = (
        (mfg_df.So*mfg_df.Km*mfg_df.Sg*mfg_df.Kp*mfg_df.Sd) /
        (2*np.pi/NCALPER*mfg_df.Rc*mfg_df.M))

    # modify inventory response and tabulate NCALIB at NCALPER
    old_inventory = client.get_stations(
        network=NETWORK,
        station=mfg_df.index[
            mfg_df.index.str.startswith(STATION_PREFIX)].str.cat(sep=','),
        endafter=END,
        level='response')

    logger.info('Saving: %s', OLD_XML)
    old_inventory.write(OLD_XML, format='StationXML')

    new_inventory = deepcopy(old_inventory)
    mfg_df['NCALIB'] = np.nan
    mfg_df['gain_expected'] = np.nan
    for station_code, mfg in mfg_df.iterrows():

        pole = (-2*np.pi*mfg.f0*(mfg.lambda0 - 1j*np.sqrt(1 - mfg.lambda0**2)))
        pole = (round_sig(pole.real, PRECISION) +
                1j*round_sig(pole.imag, PRECISION))

        sensor_zeros = [0, 0]
        sensor_poles = [pole, pole.conj()]

        # manufacturer specifies motor constant which defines sensitivity at
        # high frequencies
        sensor_norm_factor, sensor_gain = stage_params(
            sensor_zeros, sensor_poles, mfg.Sg, 1/NCALPER, 10**(PRECISION + 1))

        sensor = PolesZerosResponseStage(
            stage_sequence_number=1,
            stage_gain=round_sig(sensor_gain, PRECISION),
            stage_gain_frequency=1/NCALPER,
            input_units='m/s',
            output_units='V',
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_frequency=1/NCALPER,
            zeros=sensor_zeros,
            poles=sensor_poles,
            normalization_factor=round_sig(sensor_norm_factor, PRECISION),
            name=f'Geotech|Sensor Model {mfg.sensor_model}',
            input_units_description='velocity',
            output_units_description='voltage',
            description=(f'S/N {mfg.sensor_id} with {mfg.Rd} ohm damping '
                         'resistor'))

        preamp_poles = [-round_sig(2*np.pi*67, PRECISION)]

        preamp = PolesZerosResponseStage(
            stage_sequence_number=1,
            stage_gain=round_sig(mfg.Kp, PRECISION),
            stage_gain_frequency=0,
            input_units='V',
            output_units='V',
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_frequency=0,
            zeros=[],
            poles=preamp_poles,
            normalization_factor=-preamp_poles[0],
            name=f'Guralp|Preamp Model {mfg.preamp_model}',
            input_units_description='voltage',
            output_units_description='voltage',
            description=f'S/N {mfg.preamp_id}')

        datalogger_stages = deepcopy(nrl_datalogger_stages)

        datalogger_stages[0].stage_gain = mfg.Kd
        datalogger_stages[0].input_units = 'V'  # silences warning in evalresp
        datalogger_stages[0].output_units = 'V'
        datalogger_stages[1].stage_gain = round_sig(1/mfg.Si, PRECISION)
        datalogger_stages[1].description = f'S/N {mfg.digitizer_id}'
        datalogger_stages[1].name = \
            f'Guralp|Datalogger Model {mfg.digitizer_model}'

        response_stages = [sensor, preamp] + datalogger_stages
        for i, stage in enumerate(response_stages, start=1):
            stage.stage_sequence_number = i
        sensitivity = InstrumentSensitivity(
            value=np.prod([stage.stage_gain for stage in response_stages]),
            frequency=response_stages[0].stage_gain_frequency,
            input_units=response_stages[0].input_units,
            output_units=response_stages[-1].output_units,
            input_units_description=response_stages[0].input_units_description,
            output_units_description=(response_stages[-1]
                                      .output_units_description))
        response = Response(
            instrument_sensitivity=sensitivity,
            response_stages=response_stages)

        ncalib = 1e9/np.abs(
            response.get_evalresp_response_for_frequencies(
                [1/mfg.NCALPER], output="disp")[0])
        mfg_df.at[station_code, 'NCALIB'] = ncalib

        cal_sensor = PolesZerosResponseStage(
            stage_sequence_number=1,
            stage_gain=mfg.M/mfg.Km,
            stage_gain_frequency=0,
            input_units='m/s**2',
            output_units='A',
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_frequency=1,
            zeros=[],
            poles=[],
            normalization_factor=1,
            name=f'Geotech|Sensor Model {mfg.sensor_model}',
            input_units_description='acceleration',
            output_units_description='current',
            description=f'S/N {mfg.sensor_id}')

        cal_preamp = PolesZerosResponseStage(
            stage_sequence_number=1,
            stage_gain=(mfg.Rp + mfg.Rm)/mfg.Kc,
            stage_gain_frequency=0,
            input_units='V',
            output_units='A',
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_frequency=1,
            zeros=[],
            poles=[],
            normalization_factor=1,
            name=f'Guralp|Preamp Model {mfg.preamp_model}',
            input_units_description='voltage',
            output_units_description='current',
            description=f'S/N {mfg.preamp_id}')

        cal_datalogger_stages = deepcopy(datalogger_stages)
        cal_datalogger_stages[0].stage_gain = 1
        cal_datalogger_stages[1].stage_gain = round_sig(1/mfg.So, PRECISION)

        cal_stages = [cal_sensor, cal_preamp] + cal_datalogger_stages
        for i, stage in enumerate(cal_stages, start=1):
            stage.stage_sequence_number = i

        cal_sensitivity = InstrumentSensitivity(
            value=np.prod([stage.stage_gain for stage in cal_stages]),
            frequency=cal_stages[0].stage_gain_frequency,
            input_units=cal_stages[0].input_units,
            output_units=cal_stages[-1].output_units,
            input_units_description=(
                cal_stages[0].input_units_description),
            output_units_description=(cal_stages[-1]
                                      .output_units_description))
        cal_response = Response(
            instrument_sensitivity=cal_sensitivity,
            response_stages=cal_stages)

        # work around https://github.com/obspy/obspy/issues/3262
        cal_response_copy = deepcopy(cal_response)
        for stage in cal_response_copy.response_stages:
            stage.input_units = stage.input_units.replace('A', 'V')
            stage.output_units = stage.output_units.replace('A', 'V')
        ncalib_cal = 1e9/np.abs(
            cal_response_copy.get_evalresp_response_for_frequencies(
                [1/mfg.NCALPER], output="disp")[0])
        mfg_df.at[station_code, 'gain_expected'] = ncalib / ncalib_cal

        try:
            network, station, channel = next(
                items for items in inventory_items(new_inventory)
                if items[1].code == station_code)
        except StopIteration:
            channel = new_inventory[0][0][0].copy()
            channel.response = response
            channel.latitude = 0
            channel.longitude = 0
            channel.elevation = 0
            channel.depth = 0
            channel.description = 'Dummy Channel'
            channel.start_date = None
            channel.code = 'SHZ'
            channel.comments = [Comment(
                value=dumps({'NCALIB': round_sig(ncalib, PRECISION),
                             'NCALPER': mfg.NCALPER}),
                subject='IDC nominal response',
                authors=[AUTHOR])]

            cal_channel = channel.copy()
            cal_channel.response = cal_response
            cal_channel.code = channel.code[:2] + 'C'
            cal_channel.comments = [Comment(
                value=dumps({'NCALIB': round_sig(ncalib_cal, PRECISION),
                             'NCALPER': mfg.NCALPER}),
                subject='IDC nominal response',
                authors=[AUTHOR])]

            station = new_inventory[0][0].copy()
            station.code = str(station_code).upper()[:5]
            station.latitude = 0
            station.longitude = 0
            station.elevation = 0
            station.site.name = None
            station.site.description = 'Dummy Site'
            station.start_date = None
            station.channels = [channel, cal_channel]

            network = new_inventory[0].copy()
            network.description = 'CHIS Internal'
            network.stations = [station]
            network.code = 'XX'

            inventory = Inventory(
                networks=[network],
                source='Canadian Hazards Information Service')
            partial_xml = network.code + '.' + station.code + '.xml'
            inventory.write(partial_xml, format='StationXML')
            logger.info('Writing: %s', partial_xml)
            continue

        logger.info('Updating: %s', station_code)
        channel.response = response

    logger.info('Saving: %s', NEW_XML)
    new_inventory.write(NEW_XML, format='StationXML')

    # write manufacturer's data plus derived information
    logger.info('Saving: %s', MANUFACTURER_PLUS_FILE)
    mfg_df.to_csv(MANUFACTURER_PLUS_FILE, float_format='%.6g')
