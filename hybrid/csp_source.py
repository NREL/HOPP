from typing import Optional, Union, Sequence

import rapidjson                # NOTE: install 'python-rapidjson' NOT 'rapidjson'

import pandas as pd
import numpy as np
import datetime
import os

from hybrid.pySSC_daotk.ssc_wrap import ssc_wrap
import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch
from hybrid.power_source import *
from hybrid.sites import SiteInfo


class CspOutputs:
    """Object for storing CSP outputs from SSC (SAM's Simulation Core) and dispatch optimization."""
    def __init__(self):
        self.ssc_time_series = {}
        self.dispatch = {}

    def update_from_ssc_output(self, ssc_outputs: dict, skip_hr_start: int = 0, skip_hr_end: int = 0):
        """
        Updates stored outputs based on SSC's output dictionary.

        :param ssc_outputs: dict, SSC's output dictionary containing the previous simulation results
        :param skip_hr_start: (optional) Hours to skip at beginning of simulated array
        :param skip_hr_end: (optional) Hours to skip at end of simulated array
        """
        seconds_per_step = int(3600/ssc_outputs['time_steps_per_hour'])
        ntot = int(ssc_outputs['time_steps_per_hour'] * 8760)
        is_empty = (len(self.ssc_time_series) == 0)

        # Index in annual array corresponding to first simulated time point
        i = int(ssc_outputs['time_start'] / seconds_per_step)
        # Number of simulated time steps
        n = int((ssc_outputs['time_stop'] - ssc_outputs['time_start'])/seconds_per_step)
        s1 = int(skip_hr_start*3600/seconds_per_step)     # Time steps to skip at beginning of simulated array
        s2 = int(skip_hr_end*3600/seconds_per_step)       # Time steps to skip at end of simulated array
        i += s1
        n -= (s1+s2)  

        if is_empty:
            for name, val in ssc_outputs.items():
                if isinstance(val, list) and len(val) == ntot:  
                    self.ssc_time_series[name] = [0.0]*ntot
        
        for name in self.ssc_time_series.keys():
            self.ssc_time_series[name][i:i+n] = ssc_outputs[name][s1:s1+n]

    def store_dispatch_outputs(self, dispatch: CspDispatch, n_periods: int, sim_start_time: int):
        """
        Stores dispatch model outputs for post-processing analysis.

        :param dispatch: CSP dispatch objective with attributes to store
        :param n_periods: Number of periods to store dispatch outputs
        :param sim_start_time: The first simulation hour of the dispatch horizon
        """
        outputs_keys = ['available_thermal_generation', 'cycle_ambient_efficiency_correction', 'condenser_losses',
                        'thermal_energy_storage', 'receiver_startup_inventory', 'receiver_thermal_power',
                        'receiver_startup_consumption', 'is_field_generating', 'is_field_starting', 'incur_field_start',
                        'cycle_startup_inventory', 'system_load', 'cycle_generation', 'cycle_thermal_ramp',
                        'cycle_thermal_power', 'is_cycle_generating', 'is_cycle_starting', 'incur_cycle_start']

        is_empty = (len(self.dispatch) == 0)
        if is_empty:
            for key in outputs_keys:
                self.dispatch[key] = [0.0] * 8760

        for key in outputs_keys:
            self.dispatch[key][sim_start_time: sim_start_time + n_periods] = getattr(dispatch, key)[0: n_periods]


class CspPlant(PowerSource):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    # _layout: TroughLayout
    _dispatch: CspDispatch

    param_files: dict
    """Files contain default SSC parameter values"""

    def __init__(self,
                 name: str,
                 tech_name: str,
                 site,
                 financial_model,
                 csp_config: dict):
        """
        Abstract class for CSP technologies.

        :param name: Name used to identify technology
        :param tech_name: PySSC technology name [tcsmolten_salt, trough_physical]
        :param site: Power source site information (SiteInfo object)
        :param financial_model: Financial model for the specific technology
        :param csp_config: CSP configuration with the following keys:

            #. ``cycle_capacity_kw``: float, Power cycle  design turbine gross output [kWe]
            #. ``solar_multiple``: float, Solar multiple [-]
            #. ``tes_hours``: float, Full load hours of thermal energy storage [hrs]
        """

        required_keys = ['cycle_capacity_kw', 'solar_multiple', 'tes_hours']
        if any(key not in csp_config.keys() for key in required_keys):
            is_missing = [key not in csp_config.keys() for key in required_keys]
            missing_keys = [missed_key for (missed_key, missing) in zip(required_keys, is_missing) if missing]
            raise ValueError(type(self).__name__ + " requires the following keys: " + str(missing_keys))

        super().__init__("TowerPlant", site, None, financial_model)

        # TODO: Should 'SSC' object be a protected attr
        # Initialize ssc and get weather data
        self.ssc = ssc_wrap(
            wrapper='pyssc',  # ['pyssc' | 'pysam']
            tech_name=tech_name,  # ['tcsmolten_salt' | 'trough_physical]
            financial_name=None,
            defaults_name=None)  # ['MSPTSingleOwner' | 'PhysicalTroughSingleOwner']  NOTE: not used for pyssc
        self.initialize_params()

        self.year_weather_df = self.tmy3_to_df()  # read entire weather file

        self.cycle_capacity_kw: float = csp_config['cycle_capacity_kw']
        self.solar_multiple: float = csp_config['solar_multiple']
        self.tes_hours: float = csp_config['tes_hours']

        # Set full annual weather data once
        self.set_weather(self.year_weather_df)

        # Data for dispatch model
        self.solar_thermal_resource = list
        self.cycle_efficiency_tables = dict

        self.plant_state = self.set_initial_plant_state()
        self.update_ssc_inputs_from_plant_state()

        self.outputs = CspOutputs()

    def param_file_paths(self, relative_path: str):
        """
        Converts relative paths to absolute for files containing SSC default parameters

        :param relative_path: Relative path to data files
        """
        cwd = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(cwd, relative_path)
        for key in self.param_files.keys():
            filename = self.param_files[key]
            self.param_files[key] = os.path.join(data_path, filename)

    def initialize_params(self):
        """
        Initializes SSC parameters using default values stored in files.
        """
        self.set_params_from_files()
        self.ssc.set({'time_steps_per_hour': 1}) 
        n_steps_year = int(8760 * self.ssc.get('time_steps_per_hour'))
        self.ssc.set({'sf_adjust:hourly': n_steps_year * [0]})

        if len(self.site.elec_prices.data) == n_steps_year: 
            self.ssc.set({'ppa_multiplier_model': 1, 'dispatch_factors_ts': self.site.elec_prices.data})
        else:
            raise ValueError('Electricity prices have not been set correctly in SiteInfo.')

    def tmy3_to_df(self):
        """
        Parses TMY3 solar resource file (from SiteInfo) and coverts data to a Pandas DataFrame

        .. note::
            Be careful of leading spaces in the column names, they are hard to catch and break the parser

        :returns: Weather file data (DataFrame)
        """
        df = pd.read_csv(self.site.solar_resource.filename, sep=',', skiprows=2, header=0)
        date_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
        df.index = pd.to_datetime(df[date_cols])
        df.index.name = 'datetime'
        df.drop(date_cols, axis=1, inplace=True)

        df.index = df.index.map(lambda t: t.replace(year=df.index[0].year))  # normalize all years to that of 1/1
        df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]  # drop unnamed columns (which are empty)

        def get_weatherfile_location(tmy3_path):
            df_meta = pd.read_csv(tmy3_path, sep=',', header=0, nrows=1)
            return {
                'latitude': float(df_meta['Latitude'][0]),
                'longitude': float(df_meta['Longitude'][0]),
                'timezone': int(df_meta['Time Zone'][0]),
                'elevation': float(df_meta['Elevation'][0])
            }

        location = get_weatherfile_location(self.site.solar_resource.filename)
        df.attrs.update(location)
        return df

    def set_params_from_files(self):
        """
        Loads default case parameters from files
        """
        with open(self.param_files['tech_model_params_path'], 'r') as f:
            ssc_params = rapidjson.load(f)
        self.ssc.set(ssc_params)

        wlim_series = np.array(pd.read_csv(self.param_files['wlim_series_path']))
        self.ssc.set({'wlim_series': wlim_series})

    def set_weather(self, weather_df: pd.DataFrame, start_datetime=None, end_datetime=None):
        """
        Sets 'solar_resource_data' for pySSC simulation. If start and end (datetime) are not provided, full year is
        assumed.

        :param weather_df: weather information (DataFrame)
        :param start_datetime: start of pySSC simulation (datetime)
        :param end_datetime: end of pySSC simulation (datetime)
        """
        weather_timedelta = weather_df.index[1] - weather_df.index[0]
        weather_time_steps_per_hour = int(1 / (weather_timedelta.total_seconds() / 3600))
        ssc_time_steps_per_hour = self.ssc.get('time_steps_per_hour')
        if weather_time_steps_per_hour != ssc_time_steps_per_hour:
            raise Exception('Configured time_steps_per_hour ({x}) is not that of weather file ({y})'.format(
                x=ssc_time_steps_per_hour, y=weather_time_steps_per_hour))

        if start_datetime is None and end_datetime is None:
            if len(weather_df) != ssc_time_steps_per_hour * 8760:
                raise Exception('Full year weather dataframe required if start and end datetime are not provided')
            weather_df_part = weather_df
        else:
            weather_year = weather_df.index[0].year
            if start_datetime.year != weather_year:
                print('Replacing start and end years ({x}) with weather file\'s ({y}).'.format(
                    x=start_datetime.year, y=weather_year))
                start_datetime = start_datetime.replace(year=weather_year)
                end_datetime = end_datetime.replace(year=weather_year)

            if start_datetime < weather_df.index[0]:
                start_datetime = weather_df.index[0]

            if end_datetime <= start_datetime:
                end_datetime = start_datetime + weather_timedelta

            # times in weather file are the start (or middle) of time step
            weather_df_part = weather_df[start_datetime:(end_datetime - weather_timedelta)]

        def weather_df_to_ssc_table(weather_df):
            rename_from_to = {
                'Tdry': 'Temperature',
                'Tdew': 'Dew Point',
                'RH': 'Relative Humidity',
                'Pres': 'Pressure',
                'Wspd': 'Wind Speed',
                'Wdir': 'Wind Direction'
            }
            weather_df = weather_df.rename(columns=rename_from_to)

            solar_resource_data = {}
            solar_resource_data['tz'] = weather_df.attrs['timezone']
            solar_resource_data['elev'] = weather_df.attrs['elevation']
            solar_resource_data['lat'] = weather_df.attrs['latitude']
            solar_resource_data['lon'] = weather_df.attrs['longitude']
            solar_resource_data['year'] = list(weather_df.index.year)
            solar_resource_data['month'] = list(weather_df.index.month)
            solar_resource_data['day'] = list(weather_df.index.day)
            solar_resource_data['hour'] = list(weather_df.index.hour)
            solar_resource_data['minute'] = list(weather_df.index.minute)
            solar_resource_data['dn'] = list(weather_df['DNI'])
            solar_resource_data['df'] = list(weather_df['DHI'])
            solar_resource_data['gh'] = list(weather_df['GHI'])
            solar_resource_data['wspd'] = list(weather_df['Wind Speed'])
            solar_resource_data['tdry'] = list(weather_df['Temperature'])
            solar_resource_data['pres'] = list(weather_df['Pressure'])
            if 'Dew Point' in weather_df.columns:
                solar_resource_data['tdew'] = list(weather_df['Dew Point'])
            elif 'Relative Humidity' in weather_df.columns:
                solar_resource_data['rh'] = list(weather_df['Relative Humidity'])
            else:
                raise ValueError("CSP model requires either Dew Point or Relative Humidity "
                                 "to be specified in weather data.")

            def pad_solar_resource_data(solar_resource_data):
                datetime_start = datetime.datetime(
                    year=solar_resource_data['year'][0],
                    month=solar_resource_data['month'][0],
                    day=solar_resource_data['day'][0],
                    hour=solar_resource_data['hour'][0],
                    minute=solar_resource_data['minute'][0])
                n = len(solar_resource_data['dn'])
                if n < 2:
                    timestep = datetime.timedelta(hours=1)  # assume 1 so minimum of 8760 results
                else:
                    datetime_second_time = datetime.datetime(
                        year=solar_resource_data['year'][1],
                        month=solar_resource_data['month'][1],
                        day=solar_resource_data['day'][1],
                        hour=solar_resource_data['hour'][1],
                        minute=solar_resource_data['minute'][1])
                    timestep = datetime_second_time - datetime_start
                steps_per_hour = int(3600 / timestep.seconds)
                # Substitute a non-leap year (2009) to keep multiple of 8760 assumption:
                i0 = int((datetime_start.replace(year=2009) - datetime.datetime(2009, 1, 1, 0, 0,
                                                                                0)).total_seconds() / timestep.seconds)
                diff = 8760 * steps_per_hour - n
                front_padding = [0] * i0
                back_padding = [0] * (diff - i0)

                if diff > 0:
                    for k in solar_resource_data:
                        if isinstance(solar_resource_data[k], list):
                            solar_resource_data[k] = front_padding + solar_resource_data[k] + back_padding
                    return solar_resource_data
                else:
                    return solar_resource_data

            solar_resource_data = pad_solar_resource_data(solar_resource_data)
            return solar_resource_data

        self.ssc.set({'solar_resource_data': weather_df_to_ssc_table(weather_df_part)})

    @staticmethod
    def get_plant_state_io_map() -> dict:
        """Gets CSP plant state inputs (initial state) and outputs (last state) variables

        :returns: Dictionary with the key-value pairs correspond to inputs and outputs, respectively"""
        raise NotImplementedError

    def set_initial_plant_state(self) -> dict:
        """
        Sets CSP plant initial state based on SSC initial conditions.

        .. note::
            This assumes the receiver and the power cycle are initially off

        :returns: Dictionary containing plant state variables to be set in SSC
        """
        io_map = self.get_plant_state_io_map()
        plant_state = {k: 0 for k in io_map.keys()}
        plant_state['rec_op_mode_initial'] = 0  # Receiver initially off
        plant_state['pc_op_mode_initial'] = 3  # Cycle initially off
        plant_state['pc_startup_time_remain_init'] = self.ssc.get('startup_time')
        plant_state['pc_startup_energy_remain_initial'] = self.ssc.get('startup_frac')*self.cycle_thermal_rating*1000.  # kWh
        plant_state['sim_time_at_last_update'] = 0.0
        plant_state['T_tank_cold_init'] = self.htf_cold_design_temperature
        plant_state['T_tank_hot_init'] = self.htf_hot_design_temperature
        return plant_state

    def set_tes_soc(self, charge_percent: float) -> float:
        """Sets CSP plant TES state-of-charge

        :param charge_percent: Initial fraction of available volume that is hot [%]
        """
        raise NotImplementedError   

    def set_cycle_state(self, is_on: bool = True):
        """Sets cycle initial state

        :param is_on: True if cycle is initially on, False otherwise
        """
        self.plant_state['pc_op_mode_initial'] = 1 if is_on else 3
        if self.plant_state['pc_op_mode_initial'] == 1:
            self.plant_state['pc_startup_time_remain_init'] = 0.0
            self.plant_state['pc_startup_energy_remain_initial'] = 0.0
    
    def set_cycle_load(self, load_fraction: float):
        """Sets cycle initial thermal loading

        :param load_fraction: Thermal loading normalized by cycle thermal rating [-]
        """
        self.plant_state['heat_into_cycle'] = load_fraction * self.cycle_thermal_rating

    def get_tes_soc(self, time_hr: int) -> float:
        """Gets TES state-of-charge percentage at a specified time.

        :param time_hr: Hour in SSC simulation to get TES state-of-charge

        :returns: TES state-of-charge percentage [%]
        """
        i = int(time_hr * self.ssc.get('time_steps_per_hour'))
        tes_charge = self.outputs.ssc_time_series['e_ch_tes'][i]
        return (tes_charge / self.tes_capacity) * 100
    
    def get_cycle_load(self, time_hr: int) -> float:
        """ Gets cycle thermal loading at a specified time.

        :param time_hr: Hour in SSC simulation to get cycle thermal loading

        :returns: Cycle thermal loading normalized by cycle thermal rating [-]
        """
        i = int(time_hr * self.ssc.get('time_steps_per_hour'))
        return self.outputs.ssc_time_series['q_pb'][i] / self.cycle_thermal_rating

    def set_plant_state_from_ssc_outputs(self, ssc_outputs: dict, seconds_relative_to_start: int):
        """
        Sets CSP plant state variables based on SSC outputs dictionary

        :param ssc_outputs: dict, SSC's output dictionary containing the previous simulation results
        :param seconds_relative_to_start: Seconds relative to SSC simulation start to get CSP plant states
        """
        time_steps_per_hour = self.ssc.get('time_steps_per_hour')
        time_start = self.ssc.get('time_start')
        # Note: values returned in ssc_outputs are at the front of the output arrays
        idx = round(seconds_relative_to_start/3600) * int(time_steps_per_hour) - 1
        io_map = self.get_plant_state_io_map()
        for ssc_input, output in io_map.items():
            if ssc_input == 'T_out_scas_initial':
                self.plant_state[ssc_input] = ssc_outputs[output]
            else:
                self.plant_state[ssc_input] = ssc_outputs[output][idx]
        # Track time at which plant state was last updated
        self.plant_state['sim_time_at_last_update'] = time_start + seconds_relative_to_start

    def update_ssc_inputs_from_plant_state(self):
        """
        Updates SSC inputs from CSP plant state attribute
        """
        state = self.plant_state.copy()
        state.pop('sim_time_at_last_update')
        state.pop('heat_into_cycle')
        self.ssc.set(state)

    def setup_performance_model(self):
        """
        Runs a year long forecasting simulation of csp thermal generation, then sets power cycle efficiency tables and
        solar thermal resource for the dispatch model.
        """
        ssc_outputs = self.run_year_for_max_thermal_gen()
        self.set_cycle_efficiency_tables(ssc_outputs)
        self.set_solar_thermal_resource(ssc_outputs)

    def run_year_for_max_thermal_gen(self):
        """
        Call PySSC to estimate solar thermal resource for the whole year for dispatch model

        .. note::
            Solar field production is "forecasted" by setting TES hours to 100 and receiver start-up time
            and energy to very small values.

        :returns: ssc_outputs: dict, SSC's output dictionary containing the previous simulation results
        """
        self.value('is_dispatch_targets',  0)
        # Setting simulation times and simulate the horizon
        self.value('time_start', 0)
        self.value('time_stop', 8760*60*60)

        # Inflate TES capacity, set near-zero startup requirements, and run ssc estimates
        original_values = {k: self.ssc.get(k) for k in ['tshours', 'rec_su_delay', 'rec_qf_delay']}
        self.ssc.set({'tshours': 100, 'rec_su_delay': 0.001, 'rec_qf_delay': 0.001})
        print("Forecasting CSP thermal energy production...")
        ssc_outputs = self.ssc.execute()
        self.ssc.set(original_values)

        return ssc_outputs

    def set_cycle_efficiency_tables(self, ssc_outputs):
        """
        Sets cycle off-design performance tables from PySSC outputs.

        :params ssc_outputs: ssc_outputs: dict, SSC's output dictionary containing simulation results
        """
        required_tables = ['cycle_eff_load_table', 'cycle_eff_Tdb_table', 'cycle_wcond_Tdb_table']
        if all(table in ssc_outputs for table in required_tables):
            self.cycle_efficiency_tables = {table: ssc_outputs[table] for table in required_tables}
        elif ssc_outputs['pc_config'] == 1:
            # Tables not returned from ssc, but can be taken from user-defined cycle inputs
            self.cycle_efficiency_tables = {'ud_ind_od': ssc_outputs['ud_ind_od']}
        else:
            print('WARNING: Cycle efficiency tables not found. Dispatch optimization will assume a constant cycle '
                  'efficiency and no ambient temperature dependence.')
            self.cycle_efficiency_tables = {}

    def set_solar_thermal_resource(self, ssc_outputs):
        """
        Sets receiver estimated thermal resource using ssc outputs

        :params ssc_outputs: ssc_outputs: dict, SSC's output dictionary containing simulation results
        """
        thermal_est_name_map = {'TowerPlant': 'Q_thermal', 'TroughPlant': 'qsf_expected'}
        self.solar_thermal_resource = [max(heat, 0.0) for heat in ssc_outputs[thermal_est_name_map[type(self).__name__]]]

    def scale_params(self, params_names: list = ['tank_heaters', 'tank_height']):
        """
        Scales absolute parameters within the CSP models when design changes. Scales TES tank heater power linearly
        with respect to TES capacity. Scales TES tank height based on TES capacity assuming a constant aspect ratio
        (height/diameter)

        :params params_names: list of parameters to be scaled
        """
        if 'tank_heaters' in params_names:
            cold_heater = 15 * (self.tes_capacity / 2791.3)  # ssc default is 15 MWe with 2791.3 MWt-hr TES capacity
            hot_heater = 30 * (self.tes_capacity / 2791.3)   # ssc default is 30 MWe with 2791.3 MWt-hr TES capacity
            self.ssc.set({'cold_tank_max_heat': cold_heater, 'hot_tank_max_heat': hot_heater})

        if 'tank_height' in params_names:
            tank_min = self.value("h_tank_min")
            # assuming a constant aspect ratio h/d
            height = ((12 - tank_min)**3 * self.tes_capacity / 2791.3)**(1/3) + tank_min
            # ssc default is 12 m with 2791.3 MWt-hr TES capacity
            self.ssc.set({'h_tank': height})

    def simulate_with_dispatch(self, n_periods: int, sim_start_time: int = None, store_outputs: bool = True):
        """
        Simulate CSP system using dispatch solution as targets

        :param n_periods: Number of hours to simulate [hrs]
        :param sim_start_time: Start hour of simulation horizon
        :param store_outputs: (optional) When *True* SSC and dispatch results are stored in CspOutputs,
                                o.w. they are not stored
        """
        # Set up start and end time of simulation
        start_datetime, end_datetime = CspDispatch.get_start_end_datetime(sim_start_time, n_periods)
        self.value('time_start', CspDispatch.seconds_since_newyear(start_datetime))
        self.value('time_stop', CspDispatch.seconds_since_newyear(end_datetime))

        self.set_dispatch_targets(n_periods)
        self.update_ssc_inputs_from_plant_state()

        results = self.simulate_power()

        # Save plant state at end of simulation
        simulation_time = (end_datetime - start_datetime).total_seconds()
        self.set_plant_state_from_ssc_outputs(results, simulation_time)

        # Save simulation output
        if store_outputs:
            self.outputs.update_from_ssc_output(results)
            self.outputs.store_dispatch_outputs(self.dispatch, n_periods, sim_start_time)

    def simulate_power(self) -> dict:
        """
        Runs CSP system model simulate

        :returns: SSC results dictionary
        """
        if not self.ssc:
            raise ValueError('SSC was not correctly setup...')

        results = self.ssc.execute()
        if not results["cmod_success"]:
            raise ValueError('PySSC simulation failed...')

        return results

    def set_dispatch_targets(self, n_periods: int):
        """Set PySSC targets using dispatch model solution.

        :param n_periods: Number of hours to simulate [hrs]
        """
        # Set targets
        dis = self.dispatch

        dispatch_targets = {'is_dispatch_targets': 1,
                            # Receiver on, startup, (or standby - NOT in dispatch currently)
                            'is_rec_su_allowed_in': [1 if (dis.is_field_generating[t] + dis.is_field_starting[t]) > 0.01
                                                     else 0 for t in range(n_periods)],
                            # Receiver standby - NOT in dispatch currently
                            'is_rec_sb_allowed_in': [0 for t in range(n_periods)],
                            # Cycle on or startup
                            'is_pc_su_allowed_in': [1 if (dis.is_cycle_generating[t] + dis.is_cycle_starting[t]) > 0.01
                                                    else 0 for t in range(n_periods)],
                            # Cycle standby - NOT in dispatch currently
                            'is_pc_sb_allowed_in': [0 for t in range(n_periods)],
                            # Cycle start up thermal power
                            'q_pc_target_su_in': [dis.allowable_cycle_startup_power if dis.is_cycle_starting[t] > 0.01
                                                  else 0.0 for t in range(n_periods)],
                            # Cycle thermal power
                            'q_pc_target_on_in': dis.cycle_thermal_power[0:n_periods]}

        # Cycle max thermal power allowed
        pc_max = [max(ctp + su, dis.maximum_cycle_thermal_power) for ctp, su in
                  zip(dis.cycle_thermal_power[0:n_periods], dispatch_targets['q_pc_target_su_in'])]
        dispatch_targets['q_pc_max_in'] = pc_max

        self.ssc.set(dispatch_targets)

    def get_design_storage_mass(self) -> float:
        """Returns active storage mass [kg]"""
        e_storage = self.tes_capacity * 1000.  # Storage capacity (kWht)
        cp = self.get_cp_htf(0.5 * (self.htf_hot_design_temperature + self.htf_cold_design_temperature)) * 1.e-3  # kJ/kg/K
        m_storage = e_storage * 3600. / cp / (self.htf_hot_design_temperature - self.htf_cold_design_temperature)
        return m_storage

    def get_cycle_design_mass_flow(self) -> float:
        """Returns CSP cycle design HTF mass flow rate"""
        q_des = self.cycle_thermal_rating  # MWt
        cp_des = self.get_cp_htf(0.5 * (self.htf_hot_design_temperature + self.htf_cold_design_temperature),
                                 is_tes=False)  # J/kg/K
        m_des = q_des * 1.e6 / (cp_des * (self.htf_hot_design_temperature - self.htf_cold_design_temperature))  # kg/s
        return m_des

    def get_cp_htf(self, tc, is_tes=True):
        """Gets fluid's specific heat at temperature

        .. Note::
            Currently, this function only supports the following fluids:

            #. Salt (60% NaNO3, 40% KNO3)
            #. Nitrate_Salt
            #. Therminol_VP1

        :param tc: fluid temperature in celsius
        :param is_tes: is this the TES fluid (true) or the field fluid (false)

        :returns: HTF specific heat at temperature TC in [J/kg/K]
        """
        #  Troughs: TES "store_fluid", Field HTF "Fluid"
        fluid_name_map = {'TowerPlant': 'rec_htf', 'TroughPlant': 'store_fluid'}
        if not is_tes:
            fluid_name_map['TroughPlant'] = 'Fluid'

        tes_fluid = self.value(fluid_name_map[type(self).__name__])

        tk = tc + 273.15
        if tes_fluid == 17:
            return (-1.0e-10 * (tk ** 3) + 2.0e-7 * (tk ** 2) + 5.0e-6 * tk + 1.4387) * 1000.  # J/kg/K
        elif tes_fluid == 18:
            return 1443. + 0.172 * (tk - 273.15)
        elif tes_fluid == 21:
            return (1.509 + 0.002496 * tc + 0.0000007888 * (tc ** 2)) * 1000.
        else:
            print('HTF %d not recognized' % tes_fluid)
            return 0.0

    def get_construction_financing_cost(self) -> float:
        """
        Calculates construction financing costs based on default SAM assumptions.

        :returns: Construction financing cost [$]
        """
        # TODO: Create a flexible function to be used by all technologies
        cf = ssc_wrap('pyssc', 'cb_construction_financing', None)
        with open(self.param_files['cf_params_path'], 'r') as f:
            params = rapidjson.load(f)
        cf.set(params)
        cf.set({'total_installed_cost': self.calculate_total_installed_cost()})
        outputs = cf.execute()
        construction_financing_cost = outputs['construction_financing_cost']
        return outputs['construction_financing_cost']

    def calculate_total_installed_cost(self) -> float:
        """
        Calculates CSP plant's total installed costs using SAM's technology specific cost calculators

        :returns: Total installed cost [$]
        """
        raise NotImplementedError

    def simulate(self, interconnect_kw: float, project_life: int = 25, skip_fin=False):
        """
        Overrides ``PowerSource`` function to ensure it cannot be called
        """
        raise NotImplementedError

    def simulate_financials(self, interconnect_kw: float, project_life: int = 25, cap_cred_avail_storage: bool = True):
        """
        Sets-up and simulates financial model for CSP plants

        :param interconnect_kw: Interconnection limit [kW]
        :param project_life: (optional) Analysis period [years]
        :param cap_cred_avail_storage: Base capacity credit on available storage (True),
                                            otherwise use only dispatched generation (False)
        """
        if project_life > 1:
            self._financial_model.Lifetime.system_use_lifetime_output = 1
        else:
            self._financial_model.Lifetime.system_use_lifetime_output = 0
        self._financial_model.FinancialParameters.analysis_period = project_life

        # TODO: avoid using ssc data here?
        nameplate_capacity_kw = self.cycle_capacity_kw * self.ssc.get('gross_net_conversion_factor')
        self._financial_model.value("system_capacity", min(nameplate_capacity_kw, interconnect_kw))
        self._financial_model.value("cp_system_nameplate", min(nameplate_capacity_kw, interconnect_kw))
        self._financial_model.value("total_installed_cost", self.calculate_total_installed_cost())
        # need to store for later grid aggregation
        self.gen_max_feasible = self.calc_gen_max_feasible_kwh(interconnect_kw, cap_cred_avail_storage)
        self.capacity_credit_percent = self.calc_capacity_credit_percent(interconnect_kw)
        
        self._financial_model.Revenue.ppa_soln_mode = 1

        if len(self.generation_profile) == self.site.n_timesteps:
            single_year_gen = self.generation_profile
            self._financial_model.SystemOutput.gen = list(single_year_gen) * project_life

            self._financial_model.SystemOutput.system_pre_curtailment_kwac = list(single_year_gen) * project_life
            self._financial_model.SystemOutput.annual_energy_pre_curtailment_ac = sum(single_year_gen)

        self._financial_model.execute(0)
        logger.info("{} simulation executed".format(str(type(self).__name__)))

    def calc_gen_max_feasible_kwh(self, interconnect_kw, cap_cred_avail_storage: bool = True) -> list:
        """
        Calculates the maximum feasible generation profile that could have occurred.

        Timesteps that include startup (or could include startup if off and counting the potential
        of any stored energy) are a complication because three operating modes could exist in the
        same timestep (off, startup, on). This makes determining how long the power block (pb) is on,
        and thus its precise max generating potential, currently undetermined.

        :param interconnect_kw: Interconnection limit [kW]
        :param cap_cred_avail_storage: bool if capacity credit should be based on available storage (true),
                                            o.w. based on generation profile only (false)

        :returns: list of floats, maximum feasible generation [kWh]
        """
        SIGMA = 1e-6

        # Verify power block startup does not span timesteps
        t_step = self.value("time_steps_per_hour")                            # [hr]
        if self.value("startup_time") > t_step:
            raise NotImplementedError("Capacity credit calculations have not been implemented \
                                      for power block startup times greater than one timestep.")

        df = pd.DataFrame()
        df['Q_pb_startup'] = [x * 1e3 for x in self.outputs.ssc_time_series["q_dot_pc_startup"]]    # [kWt]
        df['E_pb_startup'] = [x * 1e3 for x in self.outputs.ssc_time_series["q_pc_startup"]]        # [kWht]
        df['W_pb_gross'] = [x * 1e3 for x in self.outputs.ssc_time_series["P_cycle"]]               # [kWe] Always average over entire timestep
        df['E_tes'] = [x * 1e3 for x in self.outputs.ssc_time_series["e_ch_tes"]]                   # [kWht]
        df['eta_pb'] = self.outputs.ssc_time_series["eta"]                                          # [-]
        df['W_pb_net'] = [x * 1e3 for x in self.outputs.ssc_time_series["P_out_net"]]  # kWe 
        df['Q_pb'] = [x * 1e3 for x in self.outputs.ssc_time_series["q_pb"]]  # kWt 

        def power_block_state(Q_pb_startup, W_pb_gross):
            """Simplified power block operating states.

            ===========   ==========================================
            State         Condition
            ===========   ==========================================
            [off]         (startup == 0 and gross output power == 0)
            [starting]    (startup  > 0 and gross output power == 0)
            [started]     (startup  > 0 and gross output power  > 0)
            [on]          (startup == 0 and gross output power  > 0) -> on to off transition still applicable
            ===========   ==========================================

            :returns: 'off'|'starting'|'started'|'on': string
            """
            if abs(Q_pb_startup) < SIGMA and abs(W_pb_gross) < SIGMA:
                return 'off'
            elif Q_pb_startup > SIGMA and abs(W_pb_gross) < SIGMA:
                return 'starting'
            elif Q_pb_startup > SIGMA and W_pb_gross > SIGMA:
                return 'started'
            elif abs(Q_pb_startup) < SIGMA and W_pb_gross > SIGMA:
                return 'on'
            else:
                return None

        def max_feasible_kwh(row):
            """
            [off]      = E_pb_possible|t_pb_on - E_startup
            [starting] = 0
            [started]  = E_pb_possible|t_pb_on
            [on]       = E_pb_possible|t_step

            :param row: Pandas Series of a row from main dataframe

            :returns: maximum feasible energy from power block [kWhe]
            """
            state = power_block_state(row.Q_pb_startup, row.W_pb_gross)

            if state == 'starting':
                return 0    # [kWhe]

            # 1. What's the maximum the power block could generate with unlimited resource, outside of startup time?
            if state == 'off':
                t_pb_startup = self.value("startup_time")                 # [hr]
            elif state == 'started':
                #t_pb_startup = row.E_pb_startup / row.Q_pb_startup \
                #                if row.E_pb_startup > SIGMA and \
                #                row.Q_pb_startup > SIGMA \
                #                else 0  #TODO: reported q_dot_pc_startup is timestep average so t_pb_startup = t_step from this calculation

                t_pb_startup  = t_step * (1.0 - row.eta_pb / (row.W_pb_gross / (row.Q_pb - row.Q_pb_startup)))  \
                                if row.E_pb_startup > SIGMA and row.Q_pb_startup > SIGMA \
                                else 0    # Fraction of timestep used for startup = 1.0 - (timestep-averaged efficiency / instantaneous efficiency while on)             

            elif state == 'on':
                t_pb_startup = 0
            else:
                return None
            W_pb_nom = self.cycle_capacity_kw                                           # [kWe]
            f_pb_max = self.value("cycle_max_frac")                       # [-]
            W_pb_max = W_pb_nom * f_pb_max                                              # [kWe]
            E_pb_max = max(W_pb_max * (t_step - t_pb_startup), row.W_pb_gross * t_step)  # [kWhe]  

            # 2. What did the power block actually generate?
            if state == 'off':
                E_pb_gross = 0                                                          # [kWhe]
            elif state == 'started' or state == 'on':
                E_pb_gross = row.W_pb_gross * t_step                                    # [kWhe] W_pb_gross avg over entire timestep
            else:
                return None

            # 3. What more could the power block generate if it used all the remaining TES (with no physical constraints)?
            if state == 'off':
                eta_pb_nom = self.cycle_nominal_efficiency                     # [-]
                f_pb_startup_of_nominal = self.value("startup_frac")      # [-]
                E_pb_startup = W_pb_nom / eta_pb_nom * f_pb_startup_of_nominal * t_pb_startup  # [kWht]
                dE_pb_rest_of_tes = max(0, row.E_tes - E_pb_startup) * eta_pb_nom              # [kWht]
            elif state == 'started' or state == 'on':
                dE_pb_rest_of_tes = row.E_tes * row.eta_pb                              # [kWhe]
            else:
                return None

            # 4. Thus, what could the power block have generated if it utilized more TES?
            E_pb_gross_max_feasible = min(E_pb_max, E_pb_gross + dE_pb_rest_of_tes)     # [kWhe]
            return E_pb_gross_max_feasible

        if cap_cred_avail_storage:
            E_pb_max_feasible = np.maximum(df['W_pb_net']*t_step, df.apply(max_feasible_kwh, axis=1)*self.value('gross_net_conversion_factor')) # [kWhe]
        else:
            E_pb_max_feasible = df['W_pb_net']*t_step 

        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        E_pb_max_feasible = np.minimum(E_pb_max_feasible, W_ac_nom*t_step)  # Limit to nominal capacity here, to avoid discrepancies between single-technology and hybrid capacity credits

        return list(E_pb_max_feasible)

    def value(self, var_name, var_value=None):
        """
        Overrides ``PowerSource.value`` to enable the use of PySSC rather than PySAM. Method looks in system model
        (PySSC) first. If unsuccessful, then it looks in the financial model  (PySAM).

        .. note::

            If system and financial models contain a variable with the same name, only the system model variable will
            be set.

        ``value(var_name)`` Gets variable value

        ``value(var_name, var_value)`` Sets variable value

        :param var_name: PySSC or PySAM variable name
        :param var_value: (optional) PySAM variable value

        :returns: Variable value (when getter)
        """
        attr_obj = None
        ssc_value = None
        if var_name in self.__dir__():
            attr_obj = self
        if not attr_obj:
            for a in self._financial_model.__dir__():
                group_obj = getattr(self._financial_model, a)
                try:
                    if var_name in group_obj.__dir__():
                        attr_obj = group_obj
                        break
                except:
                    pass
        if not attr_obj:
            try:
                ssc_value = self.ssc.get(var_name)
                attr_obj = self.ssc
            except:
                pass
        if not attr_obj:
            raise ValueError("Variable {} not found in technology or financial model {}".format(
                var_name, self.__class__.__name__))

        if var_value is None:
            if ssc_value is None:
                return getattr(attr_obj, var_name)
            else:
                return ssc_value
        else:
            try:
                if ssc_value is None:
                    setattr(attr_obj, var_name, var_value)
                else:
                    self.ssc.set({var_name: var_value})
            except Exception as e:
                raise IOError(f"{self.__class__}'s attribute {var_name} could not be set to {var_value}: {e}")

    @property
    def _system_model(self):
        """Used for dispatch to mimic other dispatch class building in hybrid dispatch builder"""
        return self

    @_system_model.setter
    def _system_model(self, value):
        pass

    @property
    def system_capacity_kw(self) -> float:
        """Gross power cycle design rating [kWe]"""
        return self.cycle_capacity_kw

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self.cycle_capacity_kw = size_kw

    @property
    def cycle_capacity_kw(self) -> float:
        """Gross power cycle design rating [kWe]"""
        return self.ssc.get('P_ref') * 1000.

    @cycle_capacity_kw.setter
    def cycle_capacity_kw(self, size_kw: float):
        self.ssc.set({'P_ref': size_kw / 1000.})

    @property
    def solar_multiple(self) -> float:
        """Solar field thermal rating over the cycle thermal rating (design conditions) [-]"""
        raise NotImplementedError

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        raise NotImplementedError

    @property
    def tes_hours(self) -> float:
        """Equivalent full-load thermal storage hours [hr]"""
        return self.ssc.get('tshours')

    @tes_hours.setter
    def tes_hours(self, tes_hours: float):
        self.ssc.set({'tshours': tes_hours})

    @property
    def tes_capacity(self) -> float:
        """TES energy capacity [MWt-hr]"""
        return self.cycle_thermal_rating * self.tes_hours

    @property
    def cycle_thermal_rating(self) -> float:
        """Design cycle thermal rating [MWt]"""
        raise NotImplementedError

    @property
    def field_thermal_rating(self) -> float:
        """Design solar field thermal rating [MWt]"""
        raise NotImplementedError

    @property
    def cycle_nominal_efficiency(self) -> float:
        """Cycle design gross efficiency [-]"""
        raise NotImplementedError

    @property
    def number_of_reflector_units(self) -> float:
        """Number of reflector units [-]"""
        raise NotImplementedError

    @property
    def minimum_receiver_power_fraction(self) -> float:
        """Minimum receiver turn down fraction [-]"""
        raise NotImplementedError

    @property
    def field_tracking_power(self) -> float:
        """Field tracking electric power [MWe]"""
        raise NotImplementedError

    @property
    def htf_cold_design_temperature(self) -> float:
        """Cold design temperature for HTF [C]"""
        raise NotImplementedError

    @property
    def htf_hot_design_temperature(self) -> float:
        """Hot design temperature for HTF [C]"""
        raise NotImplementedError

    @property
    def initial_tes_hot_mass_fraction(self) -> float:
        """Initial thermal energy storage fraction of mass in hot tank [-]"""
        raise NotImplementedError

    #
    # Outputs
    #

    @property
    def annual_energy_kwh(self) -> float:
        if self.system_capacity_kw > 0:
            return sum(list(self.outputs.ssc_time_series['gen']))
        else:
            return 0

    @property
    def generation_profile(self) -> list:
        if self.system_capacity_kw:
            return list(self.outputs.ssc_time_series['gen'])
        else:
            return [0] * self.site.n_timesteps

    @property
    def capacity_factor(self) -> float:
        if self.system_capacity_kw > 0:
            return 100. * self.annual_energy_kwh / (self.system_capacity_kw * 8760)
        else:
            return 0
