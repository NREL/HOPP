from typing import Optional, Union, Sequence
import os
import datetime
from math import pi, log, sin

import PySAM.Singleowner as Singleowner

from hybrid.dispatch.power_sources.tower_dispatch import TowerDispatch
from hybrid.dispatch.power_sources.csp_dispatch import CspDispatch


from hybrid.power_source import *
from hybrid.csp_source import CspPlant


# TODO: Figure out where to put this...
def copydoc(fromfunc, sep="\n"):
    """
    Decorator: Copy the docstring of `fromfunc`
    """
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ == None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func
    return _decorator


class TowerPlant(CspPlant):
    _system_model: None
    _financial_model: Singleowner.Singleowner
    # _layout: TowerLayout
    _dispatch: TowerDispatch

    def __init__(self,
                 site: SiteInfo,
                 tower_config: dict):
        """
        Tower concentrating solar power class based on SSC’s MSPT (molten salt power tower) model

        :param site: Power source site information
        :param tower_config: CSP configuration with the following keys:

            #. ``cycle_capacity_kw``: float, Power cycle  design turbine gross output [kWe]
            #. ``solar_multiple``: float, Solar multiple [-]
            #. ``tes_hours``: float, Full load hours of thermal energy storage [hrs]
            #. ``optimize_field_before_sim``: (optional, default = True) bool, If True, SolarPilot's field and tower
               height optimization will before system simulation, o.w., SolarPilot will just generate field based on
               inputs.
            #. ``scale_input_params``: (optional, default = True) bool, If True, HOPP will run
               :py:func:`hybrid.tower_source.scale_params` before system simulation.
        """
        financial_model = Singleowner.default('MSPTSingleOwner')

        # set-up param file paths
        self.param_files = {'tech_model_params_path': 'tech_model_defaults.json',
                            'cf_params_path': 'construction_financing_defaults.json',
                            'wlim_series_path': 'wlim_series.csv',
                            'helio_positions_path': 'helio_positions.csv'}
        rel_path_to_param_files = os.path.join('pySSC_daotk', 'tower_data')
        self.param_file_paths(rel_path_to_param_files)

        super().__init__("TowerPlant", 'tcsmolten_salt', site, financial_model, tower_config)

        self.optimize_field_before_sim = True
        if 'optimize_field_before_sim' in tower_config:
            self.optimize_field_before_sim = tower_config['optimize_field_before_sim']

        # (optionally) adjust ssc input parameters based on tower capacity 
        self.is_scale_params = False
        if 'scale_input_params' in tower_config and tower_config['scale_input_params']:
            self.is_scale_params = True
            # Parameters to be scaled before receiver size optimization
            self.scale_params(params_names=['helio_size', 'helio_parasitics', 'tank_heaters', 'tank_height'])

        self._dispatch: TowerDispatch = None

    def set_params_from_files(self):
        super().set_params_from_files()

        # load heliostat field  # TODO: this is required but is replaced when new field is generated
        heliostat_layout = np.genfromtxt(self.param_files['helio_positions_path'], delimiter=',')
        N_hel = heliostat_layout.shape[0]
        helio_positions = [heliostat_layout[j, 0:2].tolist() for j in range(N_hel)]
        self.ssc.set({'helio_positions': helio_positions})

    def scale_params(self, params_names: list = ['tank_heaters', 'tank_height']):
        """
        Scales SSC MSPT input parameters that don't automatically scale with plant capacity.

        :param params_names: list of parameters to be scaled. The following variable scaling is supported:

        ====================  =====================================  =========================================
        Parameter Names       Definition                             Method
        ====================  =====================================  =========================================
        ``tank_heaters``      Rated heater capacity for TES tanks    linearly with respect to TES capacity
        ``tank_height``       Height of HTF when tank is full        constant aspect ratio (height/diameter)
        ``helio_size``        Heliostat height and width             approximating based on a approx. receiver
        ``helio_parasitics``  Heliostat parasitic power              linearly with respect to heliostat area
        ``tube_size``         Receiver tube diameter                 scaled for specified target velocity
        ====================  =====================================  =========================================
        """
        super().scale_params(params_names)

        # Heliostat size
        if 'helio_size' in params_names:
            # Allowable upper bound (m) for heliostat size (likely ~16m is reasonable, constraining to 12.2 to
            # avoid scaling default case)
            helio_size_limit = 16
            # rough approximation, based on default case at solar noon on the summer solstice
            avg_to_peak_flux_ratio = 0.65
            f = 0.7  # Ratio of heliostat size to approximate receiver height or diameter
            rec_area_approx = self.field_thermal_rating * 1000 / (self.ssc.get('flux_max') * avg_to_peak_flux_ratio)
            rec_height_approx = (rec_area_approx / 3.14159)**0.5  # Receiver dimension for equal height/diameter
            helio_dimension = min(f * rec_height_approx, helio_size_limit)
            self.ssc.set({'helio_width': helio_dimension, 'helio_height': helio_dimension})

        # Heliostat startup and tracking power (scaled linearly based on heliostat area relative to default case)
        if 'helio_parasitics' in params_names:
            helio_area = self.value('helio_width')*self.value('helio_height')
            helio_startup_energy = 0.025 * (helio_area / 12.2**2)  # ssc default is 0.025 kWhe with 12.2x12.2m heliostats
            helio_tracking_power = 0.055 * (helio_area / 12.2**2)  # ssc default is 0.055 kWe with 12.2x12.2m heliostats
            self.ssc.set({'p_start': helio_startup_energy, 'p_track': helio_tracking_power})

        # Tube size (scaled for specified target velocity at design point mass flow)
        if 'tube_size' in params_names:
            target_velocity = 3.5  # Target design point velocity (m/s)
            npanels = self.value('N_panels')
            twall = self.value('th_tube')/1000
            npath = 1
            if self.value('Flow_type') == 1 or self.value('Flow_type') == 2:
                npath = 2
            elif self.value('Flow_type') == 9:
                npath = int(npanels / 2)
            m_rec_design = self.get_receiver_design_mass_flow()  # kg/s
            Tavg = 0.5 * (self.value('T_htf_cold_des') + self.value('T_htf_hot_des'))
            rho = self.get_density_htf(Tavg)
            visc = self.get_visc_htf(Tavg)
            panel_width = self.value('D_rec') * sin(pi/npanels)
            a = rho * target_velocity*pi
            b = -4*m_rec_design/npath/panel_width
            c = b*2*twall
            tube_id = (-b + (b**2 - 4*a*c)**0.5) / 2 / a 
            Re = rho * target_velocity * tube_id / visc
            tube_od = tube_id + 2*twall
            self.ssc.set({'d_tube_out': tube_od*1000})

    def create_field_layout_and_simulate_flux_eta_maps(self, optimize_tower_field: bool = False):
        """
        Creates heliostats field layout and simulates receiver flux efficiency maps to be stored.

        :param optimize_tower_field: If True, SolarPilot's field and tower height optimization will before system
            simulation, o.w., SolarPilot will just generate field based on inputs.
        """
        self.ssc.set({'time_start': 0})
        self.ssc.set({'time_stop': 0})

        if optimize_tower_field:
            # Run field, tower height, and receiver diameter and height optimization
            self.ssc.set({'field_model_type': 0})
            print('Optimizing field layout, tower height, receiver diameter, and receiver height'
                  ' and simulating flux and eta maps ...')
        else:
            # Create field layout and generate flux and eta maps, but don't optimize field or tower
            self.ssc.set({'field_model_type': 1})
            print('Generating field layout and simulating flux and eta maps ...')

        original_values = {k: self.ssc.get(k) for k in['is_dispatch_targets', 'rec_clearsky_model', 'time_steps_per_hour', 'sf_adjust:hourly']}
        # set so unneeded dispatch targets and clearsky DNI are not required
        # TODO: probably don't need hourly sf adjustment factors
        self.ssc.set({'is_dispatch_targets': False, 'rec_clearsky_model': 1, 'time_steps_per_hour': 1,
                      'sf_adjust:hourly': [0.0 for j in range(8760)]})
        tech_outputs = self.ssc.execute()
        print('Finished creating field layout and simulating flux and eta maps. # Heliostats = %d, Tower height = %.1fm, Receiver height = %.2fm, Receiver diameter = %.2fm'%
             (tech_outputs['N_hel'], tech_outputs['h_tower'], tech_outputs['rec_height'], tech_outputs['D_rec']))
        self.ssc.set(original_values)
        eta_map = tech_outputs["eta_map_out"]
        flux_maps = [r[2:] for r in tech_outputs['flux_maps_for_import']]  # don't include first two columns
        A_sf_in = tech_outputs["A_sf"]
        field_and_flux_maps = {'eta_map': eta_map, 'flux_maps': flux_maps, 'A_sf_in': A_sf_in}
        for k in ['helio_positions', 'N_hel', 'D_rec', 'rec_height', 'h_tower', 'land_area_base']:
            field_and_flux_maps[k] = tech_outputs[k]

        # Check if specified receiver dimensions make sense relative to heliostat dimensions
        if min(field_and_flux_maps['rec_height'], field_and_flux_maps['D_rec']) < max(self.ssc.get('helio_width'), self.ssc.get('helio_height')):
            print('Warning: Receiver height or diameter is smaller than the heliostat dimension. Design will likely have high spillage loss. Heliostat width and height = %.2fm'%
                  (self.ssc.get('helio_width')))

        self.ssc.set(field_and_flux_maps)  # set flux maps etc. so they don't have to be recalculated
        self.ssc.set({'field_model_type': 3})  # use the provided flux and eta map inputs
        self.ssc.set({'eta_map_aod_format': False})

        if self.is_scale_params:  # Scale parameters that depend on receiver size
            self.scale_params(params_names=['tube_size'])

        return field_and_flux_maps

    def optimize_field_and_tower(self):
        """Optimizes heliostat field, tower height, and receiver geometry (diameter and height). This method uses
        SolarPILOT's internal optimization methods.

        .. note::

            We believe there is a memory leak when calling SolarPILOT's optimization routine. This is not problematic
            when running a single hybrid simulation. However, this can be a problem when iterating HOPP for
            optimization.
        """
        self.create_field_layout_and_simulate_flux_eta_maps(optimize_tower_field=True)

    def generate_field(self):
        """Generates heliostat field based on current parameter values using SolarPilot."""
        self.create_field_layout_and_simulate_flux_eta_maps()

    def setup_performance_model(self):
        """Sets up tower heliostat field then runs CSP setup function.
        
        .. note::
            Set-up functions musted be called before calculate_installed_cost()
        """
        if self.optimize_field_before_sim:
            self.optimize_field_and_tower()
        else:
            self.generate_field()

        super().setup_performance_model()

    @copydoc(CspPlant.calculate_total_installed_cost)
    def calculate_total_installed_cost(self) -> float:
        """
        .. note::
            This must be called after heliostat field layout is created
        """
        # Tower total installed cost is also a direct output from the ssc compute module
        # TODO: should we pull this directly from SSC
        site_improvement_cost = self.ssc.get('site_spec_cost') * self.ssc.get('A_sf_in')
        heliostat_cost = self.ssc.get('cost_sf_fixed') + self.ssc.get('heliostat_spec_cost') * self.ssc.get('A_sf_in')

        height = self.ssc.get('h_tower')-0.5*self.ssc.get('rec_height') + 0.5*self.ssc.get('helio_height')
        tower_cost = self.ssc.get('tower_fixed_cost') * np.exp(self.ssc.get('tower_exp') * height)
        Arec = 3.1415926 * self.ssc.get('rec_height') * self.ssc.get('D_rec')
        receiver_cost = self.ssc.get('rec_ref_cost') * (Arec / self.ssc.get('rec_ref_area'))**self.ssc.get('rec_cost_exp')
        tower_receiver_cost = tower_cost + receiver_cost

        tes_cost = self.tes_capacity * 1000 * self.ssc.get('tes_spec_cost')
        cycle_cost = self.ssc.get('P_ref') * 1000 * self.ssc.get('plant_spec_cost')
        bop_cost = self.ssc.get('P_ref') * 1000 * self.ssc.get('bop_spec_cost')
        fossil_backup_cost = self.ssc.get('P_ref') * 1000 * self.ssc.get('fossil_spec_cost')
        direct_cost = site_improvement_cost + heliostat_cost + tower_receiver_cost + tes_cost + cycle_cost + bop_cost + fossil_backup_cost
        contingency_cost = self.ssc.get('contingency_rate')/100 * direct_cost
        total_direct_cost = direct_cost + contingency_cost
        total_land_area = self.ssc.get('land_area_base') * self.ssc.get('csp.pt.sf.land_overhead_factor') + self.ssc.get('csp.pt.sf.fixed_land_area')
        plant_net_capacity = self.ssc.get('P_ref') * self.ssc.get('gross_net_conversion_factor')
        
        land_cost = total_land_area * self.ssc.get('land_spec_cost') + \
                    total_direct_cost * self.ssc.get('csp.pt.cost.plm.percent')/100 + \
                    plant_net_capacity * 1e6 * self.ssc.get('csp.pt.cost.plm.per_watt') + \
                    self.ssc.get('csp.pt.cost.plm.fixed')

        epc_cost = total_land_area * self.ssc.get('csp.pt.cost.epc.per_acre') + \
                   total_direct_cost * self.ssc.get('csp.pt.cost.epc.percent')/100 + \
                   plant_net_capacity * 1e6 * self.ssc.get('csp.pt.cost.epc.per_watt') + \
                   self.ssc.get('csp.pt.cost.epc.fixed')
        
        sales_tax_cost = total_direct_cost * self.ssc.get('sales_tax_frac')/100 * self.ssc.get('sales_tax_rate')/100
        total_indirect_cost = land_cost + epc_cost + sales_tax_cost
        total_installed_cost = total_direct_cost + total_indirect_cost
        return total_installed_cost

    def estimate_receiver_pumping_parasitic(self, nonheated_length=0.2):
        """
        Estimates receiver pumping parasitic power for dispatch parameter

        :param nonheated_length: percentage of non-heated length for the receiver

        :returns: Receiver pumping power per thermal rating [MWe/MWt]
        """
        m_rec_design = self.get_receiver_design_mass_flow()  # kg/s
        Tavg = 0.5 * (self.value('T_htf_cold_des') + self.value('T_htf_hot_des'))
        rho = self.get_density_htf(Tavg)
        visc = self.get_visc_htf(Tavg)

        npath = 1
        nperpath = self.value('N_panels')
        if self.value('Flow_type') == 1 or self.value('Flow_type') == 2:
            npath = 2
            nperpath = int(nperpath / 2)
        elif self.value('Flow_type') == 9:
            npath = int(nperpath / 2)
            nperpath = 2

        ntube = int(pi * self.value('D_rec') / self.value('N_panels') / (
                self.value('d_tube_out') * 1.e-3))  # Number of tubes per panel
        m_per_tube = m_rec_design / npath / ntube  # kg/s per tube
        tube_id = (self.value('d_tube_out') - 2 * self.value('th_tube')) / 1000.  # Tube ID in m
        Ac = 0.25 * pi * (tube_id ** 2)
        vel = m_per_tube / rho / Ac  # HTF velocity
        Re = rho * vel * tube_id / visc
        if Re < 2300:
            print("Warning: Poor Receiver Design! Receiver will experience laminar flow. Consider revising.")
        eD = 4.6e-5 / tube_id
        ff = (-1.737 * log(0.269 * eD - 2.185 / Re * log(0.269 * eD + 14.5 / Re))) ** -2
        fd = 4 * ff
        Htot = self.value('rec_height') * (1 + nonheated_length)
        dp = 0.5 * fd * rho * (vel ** 2) * (
                Htot / tube_id + 4 * 30 + 2 * 16) * nperpath
        # Frictional pressure drop (Pa) (straight tube, 90deg bends, 45def bends)
        dp += rho * 9.8 * self.value('h_tower')  # Add pressure drop from pumping up the tower
        if nperpath % 2 == 1:
            dp += rho * 9.8 * Htot

        # Pumping parasitic at design point reciever mass flow rate (MWe)
        wdot = dp * m_rec_design / rho / self.value('eta_pump') / 1.e6
        return wdot / self.field_thermal_rating  # MWe / MWt

    def get_receiver_design_mass_flow(self):
        """
        Calculates receiver mass flow rate based on design temperature conditions.

        :returns: Receiver design mass flow rate [kg/s]
        """
        cp_des = self.get_cp_htf(0.5*(self.value('T_htf_cold_des') + self.value('T_htf_hot_des')))  # J/kg/K
        m_des = self.field_thermal_rating*1.e6 / (cp_des * (self.value('T_htf_hot_des')
                                                            - self.value('T_htf_cold_des')))  # kg/s
        return m_des

    def get_density_htf(self, TC):
        """Calculates HTF density based on temperature.

        .. note::
            Currently, only Salt (60% NaNO3, 40% KNO3) is supported by this function.

        :param TC: HTF temperature [C]

        :returns: HTF density [kg/m^3]
        """
        if self.value('rec_htf') != 17:
            print('HTF %d not recognized' % self.value('rec_htf'))
            return 0.0
        TK = TC+273.15
        return -1.0e-7*(TK**3) + 2.0e-4*(TK**2) - 0.7875*TK + 2299.4  # kg/m3

    def get_visc_htf(self, TC):
        """Calculates HTF viscosity based on temperature.

        .. note::
            Currently, only Salt (60% NaNO3, 40% KNO3) is supported by this function.

        :param TC: HTF temperature [C]

        :returns: HTF viscosity [kg/m-s]
        """
        if self.value('rec_htf') != 17:
            print('HTF %d not recognized' % self.value('rec_htf'))
            return 0.0
        return max(1e-4, 0.02270616 - 1.199514e-4*TC + 2.279989e-7*TC*TC - 1.473302e-10*TC*TC*TC)

    @staticmethod
    def get_plant_state_io_map() -> dict:
        io_map = {  # State:
                  # Number Inputs                         # Arrays Outputs (end of timestep)
                  'is_field_tracking_init':               'is_field_tracking_final',
                  'rec_op_mode_initial':                  'rec_op_mode_final',
                  'rec_startup_time_remain_init':         'rec_startup_time_remain_final',
                  'rec_startup_energy_remain_init':       'rec_startup_energy_remain_final',

                  'T_tank_cold_init':                     'T_tes_cold',
                  'T_tank_hot_init':                      'T_tes_hot',
                  'csp.pt.tes.init_hot_htf_percent':      'hot_tank_htf_percent_final',

                  'pc_op_mode_initial':                   'pc_op_mode_final',
                  'pc_startup_time_remain_init':          'pc_startup_time_remain_final',
                  'pc_startup_energy_remain_initial':     'pc_startup_energy_remain_final',
                  # For dispatch ramping penalty
                  'heat_into_cycle':                      'q_pb'
                  }
        return io_map        

    def set_initial_plant_state(self) -> dict:
        plant_state = super().set_initial_plant_state()
        # Use initial storage charge state that came from tech_model_defaults.json file
        plant_state['csp.pt.tes.init_hot_htf_percent'] = self.value('csp.pt.tes.init_hot_htf_percent')

        plant_state['rec_startup_time_remain_init'] = self.value('rec_su_delay')
        plant_state['rec_startup_energy_remain_init'] = (self.value('rec_qf_delay') * self.field_thermal_rating
                                                         * 1e6)  # MWh -> Wh
        return plant_state

    def set_tes_soc(self, charge_percent):
        self.plant_state['csp.pt.tes.init_hot_htf_percent'] = charge_percent

    @property
    def solar_multiple(self) -> float:
        return self.ssc.get('solarm')

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        :param solar_multiple:
        :return:
        """
        self.ssc.set({'solarm': solar_multiple})

    @property
    def cycle_thermal_rating(self) -> float:
        return self.value('P_ref') / self.value('design_eff')

    @property
    def field_thermal_rating(self) -> float:
        return self.value('solarm') * self.cycle_thermal_rating

    @property
    def cycle_nominal_efficiency(self) -> float:
        return self.value('design_eff')

    @property
    def number_of_reflector_units(self) -> float:
        """Returns number of heliostats within the field"""
        return self.value('N_hel')

    @property
    def minimum_receiver_power_fraction(self) -> float:
        """Returns minimum receiver mass flow rate turn down fraction."""
        return self.value('f_rec_min')

    @property
    def field_tracking_power(self) -> float:
        """Returns power load for field to track sun position in MWe"""
        return self.value('p_track') * self.number_of_reflector_units / 1e3

    @property
    def htf_cold_design_temperature(self) -> float:
        """Returns cold design temperature for HTF [C]"""
        return self.value('T_htf_cold_des')

    @property
    def htf_hot_design_temperature(self) -> float:
        """Returns hot design temperature for HTF [C]"""
        return self.value('T_htf_hot_des')

    @property
    def initial_tes_hot_mass_fraction(self) -> float:
        """Returns initial thermal energy storage fraction of mass in hot tank [-]"""
        return self.plant_state['csp.pt.tes.init_hot_htf_percent'] / 100.


