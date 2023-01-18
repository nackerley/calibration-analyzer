# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:19:39 2020

@author: nackerle
"""
import os
import sys
import numpy as np
from copy import deepcopy

import pandas as pd

from obspy import UTCDateTime
from obspy.core.inventory import (
    Inventory, Response,
    PolesZerosResponseStage, FIRResponseStage, InstrumentSensitivity)
from obspy.clients.fdsn import Client
from obspy.clients.nrl import NRL


from station_tools.core import CHIS_FDSN_SERVERS
from catalogue_tools.utilities import round_sig

sys.path.append('..')
from shared import (  # noqa: E402
    get_logger, TCR_CU_OHM_DEGC, ALPHA_BR_ALNICO, inventory_items)

# %% constants
current_dir = os.path.dirname(os.path.abspath(__file__))
MANUFACTURER_FILE = os.path.join(current_dir, 'CalibrationSheets.csv')
MANUFACTURER_PLUS_FILE = os.path.splitext(MANUFACTURER_FILE)[0] + 'Derived.csv'

NETWORK = 'CN'
STATION_PREFIX = 'YKA'
PRECISION = 5
START = UTCDateTime('2015-11-23 16:28:19')
END = UTCDateTime('2019-03-06 21:40:26')
STAGE_FREQUENCY = 10
NCALPER = 0.25
DATALOGGER_KEYS = ['Guralp', 'CMG-DM24', 'Mk3', 'Variable', '2', '31-40', '31',
                   '40']
SENSOR_KEYS = ['Geotech', 'Short-period sensors (GS-13, S-13, S-13J)',
               'S-13, 629 V/M/S, 3600 Ohms']

OLD_XML = '%s.%s_existing.xml' % (NETWORK, STATION_PREFIX)
NEW_XML = '%s.%s_new.xml' % (NETWORK, STATION_PREFIX)

BASE_NAME = os.path.join(current_dir,
                         os.path.basename(os.path.splitext(__file__)[0]))

if __name__ == '__main__':

    # %% initialization
    logger = get_logger(__name__, BASE_NAME + '.log')
    logger.info('Client: ' + CHIS_FDSN_SERVERS[0])
    client = Client(CHIS_FDSN_SERVERS[0])
    nrl = NRL()
    nrl_response = nrl.get_response(DATALOGGER_KEYS, SENSOR_KEYS)

    # %% load data
    logger.info('Loading: ' + MANUFACTURER_FILE)
    mfg_df = pd.read_csv(MANUFACTURER_FILE, index_col='station_code')
    nominal = mfg_df.loc['YKNOM']
    for column in mfg_df.columns:
        if isinstance(nominal[column], float) and not pd.isnull(
                nominal[column]):
            mfg_df[column].fillna(nominal[column], inplace=True)

    # %% compute derived parameters
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

    # %% modify inventory response and tabulate NCALIB at NCALPER
    old_inventory = client.get_stations(
        network=NETWORK,
        station=mfg_df.index[
            mfg_df.index.str.startswith(STATION_PREFIX)].str.cat(sep=','),
        endafter=END,
        level='response')

    logger.info('Saving: ' + OLD_XML)
    old_inventory.write(OLD_XML, format='StationXML')

    new_inventory = deepcopy(old_inventory)
    mfg_df['NCALIB'] = np.NaN
    mfg_df['gain_expected'] = np.NaN
    for station_code, mfg in mfg_df.iterrows():

        pole = (-2*np.pi*mfg.f0*(mfg.lambda0 - 1j*np.sqrt(1 - mfg.lambda0**2)))
        pole = round_sig(pole.real, PRECISION) + 1j*round_sig(pole.imag,
                                                              PRECISION)

        sensor = PolesZerosResponseStage(
            stage_sequence_number=1,
            stage_gain=1,
            stage_gain_frequency=STAGE_FREQUENCY,
            input_units='m/s',
            output_units='V',
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_frequency=STAGE_FREQUENCY,
            zeros=[0, 0],
            poles=[pole, pole.conj()],
            normalization_factor=1,
            name='Geotech|Sensor Model %s' % mfg.sensor_model,
            input_units_description='velocity',
            output_units_description='voltage',
            description=('S/N %s with %g ohm damping resistor' %
                         (mfg.sensor_id, mfg.Rd)))
        sensor_response = Response(
            response_stages=[sensor],
            instrument_sensitivity=InstrumentSensitivity(
                    value=sensor.stage_gain,
                    frequency=sensor.stage_gain_frequency,
                    input_units=sensor.input_units,
                    output_units=sensor.output_units,
                    input_units_description=sensor.input_units_description,
                    output_units_description=sensor.output_units_description))
        norm_resp = np.abs(
            sensor_response.get_evalresp_response_for_frequencies(
                np.array([sensor.normalization_frequency]).astype(float)))[0]
        sensor.stage_gain = round_sig(mfg.Sg*norm_resp, PRECISION)
        sensor.normalization_factor = round_sig(1/norm_resp)

        preamp = PolesZerosResponseStage(
            stage_sequence_number=1,
            stage_gain=mfg.Kp,
            stage_gain_frequency=0,
            input_units='V',
            output_units='V',
            pz_transfer_function_type='LAPLACE (RADIANS/SECOND)',
            normalization_frequency=1,
            zeros=[],
            poles=[],
            normalization_factor=1,
            name='Guralp|Preamp Model %s' % mfg.preamp_model,
            input_units_description='voltage',
            output_units_description='voltage',
            description='S/N %s' % mfg.preamp_id)

        datalogger_stages = deepcopy(nrl_response.response_stages[1:])
        datalogger_stages[0].stage_gain = mfg.Kd
        datalogger_stages[1].name = ('Guralp|Datalogger Model %s' %
                                     mfg.digitizer_model)
        datalogger_stages[1].stage_gain = round_sig(1/mfg.Si, PRECISION)
        datalogger_stages[1].description = 'S/N %s' % mfg.digitizer_id

        response_stages = [sensor, preamp] + datalogger_stages
        for i, stage in enumerate(response_stages, start=1):
            stage.stage_sequence_number = i
            if isinstance(stage, FIRResponseStage):
                stage.name = 'Guralp|Datalogger Model %s' % mfg.digitizer_model
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

        ncalib_resp = np.abs(response.get_evalresp_response_for_frequencies(
            np.array([1/mfg.NCALPER]).astype(float),
            start_stage=1, end_stage=1))
        gain_adjustment = ncalib_resp[0]/sensor.stage_gain
        mfg_df.at[station_code, 'NCALIB'] = mfg.NCALIB0/gain_adjustment
        mfg_df.at[station_code, 'gain_expected'] = mfg.gain0*gain_adjustment

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
            name='Geotech|Sensor Model %s' % mfg.sensor_model,
            input_units_description='acceleration',
            output_units_description='current',
            description=('S/N %s' % (mfg.sensor_id)))

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
            name='Guralp|Preamp Model %s' % mfg.preamp_model,
            input_units_description='voltage',
            output_units_description='current',
            description='S/N %s' % mfg.preamp_id)

        cal_datalogger_stages = deepcopy(nrl_response.response_stages[1:])
        cal_datalogger_stages[0].stage_gain = 1/mfg.So  # Sc in white paper
        cal_datalogger_stages[1].name = ('Guralp|Datalogger Model %s' %
                                         mfg.digitizer_model)
        cal_datalogger_stages[1].stage_gain = round_sig(1/mfg.Si, PRECISION)
        cal_datalogger_stages[1].description = 'S/N %s' % mfg.digitizer_id

        cal_response_stages = [cal_sensor, cal_preamp] + cal_datalogger_stages
        for i, stage in enumerate(cal_response_stages, start=1):
            stage.stage_sequence_number = i
            if isinstance(stage, FIRResponseStage):
                stage.name = 'Guralp|Datalogger Model %s' % mfg.digitizer_model
        cal_sensitivity = InstrumentSensitivity(
            value=np.prod([stage.stage_gain for stage in cal_response_stages]),
            frequency=cal_response_stages[0].stage_gain_frequency,
            input_units=cal_response_stages[0].input_units,
            output_units=cal_response_stages[-1].output_units,
            input_units_description=(
                cal_response_stages[0].input_units_description),
            output_units_description=(cal_response_stages[-1]
                                      .output_units_description))
        cal_response = Response(
            instrument_sensitivity=cal_sensitivity,
            response_stages=cal_response_stages)

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

            cal_channel = channel.copy()
            cal_channel.response = cal_response
            cal_channel.code = channel.code[:2] + 'C'

            station = new_inventory[0][0].copy()
            station.code = station_code.upper()[:5]
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
            logger.info('Writing: ' + partial_xml)
            continue

        logger.info('Updating: ' + station_code)
        channel.response = response

    logger.info('Saving: ' + NEW_XML)
    new_inventory.write(NEW_XML, format='StationXML')

    # %% write manufacturer's data plus derived information
    logger.info('Saving: ' + MANUFACTURER_PLUS_FILE)
    mfg_df.to_csv(MANUFACTURER_PLUS_FILE, float_format='%.6g')
