import os

from attrs import define, field
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.csp.csp_plant import CspPlant, CspConfig
from hopp.simulation.technologies.sites import SiteInfo
from hopp.utilities.validators import contains


@define
class TroughConfig(CspConfig):
    """
    Configuration class for `TroughPlant`.

    Args:
        cycle_capacity_kw: Power cycle design turbine gross output [kWe]
        solar_multiple: Solar multiple [-]
        tes_hours: Full load hours of thermal energy storage [hrs]
        fin_model: Financial model for the specific technology
    """
    tech_name: str = field(validator=contains(["tcsmolten_salt", "trough_physical"]), default="trough_physical")
    name: str = field(default="TroughPlant")


@define
class TroughPlant(CspPlant):
    """
    Parabolic trough concentrating solar power class based on SSC's Parabolic trough (physical model).

    Args:
        site: Power source site information
        config: CSP configuration
    """
    site: SiteInfo
    config: TroughConfig

    def __attrs_post_init__(self):
        if self.config.fin_model is None:
            self.config.fin_model = Singleowner.default('PhysicalTroughSingleOwner')

        # set-up param file paths
        # TODO: Site should have dispatch factors consistent across all models
        self.param_files = {'tech_model_params_path': 'tech_model_defaults.json',
                            'cf_params_path': 'construction_financing_defaults.json',
                            'wlim_series_path': 'wlim_series.csv'}
        rel_path_to_param_files = os.path.join('pySSC_daotk', 'trough_data')
        self.param_file_paths(rel_path_to_param_files)

        # Run code in parent post_init
        super().__attrs_post_init__()

        self._dispatch = None

    def calculate_aperture_and_land_area(self) -> tuple:
        """
        Calculates total aperture area and total land area by executing SSC

        Returns:
            Total aperture [m^2], Total land area [acre]
        """
        self.ssc.set({'time_start': 0.0, 'time_stop': 0.0})
        self.ssc.set({'is_dispatch_targets': 0})
        tech_outputs = self.ssc.execute()
        return tech_outputs['total_aperture'], tech_outputs['total_land_area']

    def calculate_total_installed_cost(self) -> float:
        total_aperture, total_land_area = self.calculate_aperture_and_land_area()

        gross_capacity = self.ssc.get('P_ref')  # MWe
        net_capacity = self.ssc.get('P_ref') * self.ssc.get('gross_net_conversion_factor')  # MWe
        tes_capacity = gross_capacity / self.ssc.get('eta_ref')*self.ssc.get('tshours')  # MWhe

        site_improvement_cost = self.ssc.get('csp.dtr.cost.site_improvements.cost_per_m2') * total_aperture
        field_cost = self.ssc.get('csp.dtr.cost.solar_field.cost_per_m2') * total_aperture
        htf_system_cost = self.ssc.get('csp.dtr.cost.htf_system.cost_per_m2') * total_aperture
        tes_cost = tes_capacity * 1000 * self.ssc.get('csp.dtr.cost.storage.cost_per_kwht')
        cycle_cost = gross_capacity * 1000 * self.ssc.get('csp.dtr.cost.power_plant.cost_per_kwe')
        bop_cost = gross_capacity * 1000 * self.ssc.get('csp.dtr.cost.bop_per_kwe')
        fossil_backup_cost = gross_capacity * 1000 * self.ssc.get('csp.dtr.cost.fossil_backup.cost_per_kwe')
        direct_cost = site_improvement_cost + field_cost + htf_system_cost + tes_cost + cycle_cost + bop_cost + fossil_backup_cost
        contingency_cost = self.ssc.get('csp.dtr.cost.contingency_percent')/100 * direct_cost
        total_direct_cost = direct_cost + contingency_cost

        land_cost = (total_land_area * self.ssc.get('csp.dtr.cost.plm.per_acre')
                     + total_direct_cost * self.ssc.get('csp.dtr.cost.plm.percent')/100
                     + net_capacity * 1e6 * self.ssc.get('csp.dtr.cost.plm.per_watt')
                     + self.ssc.get('csp.dtr.cost.plm.fixed'))

        epc_cost = (total_land_area * self.ssc.get('csp.dtr.cost.epc.per_acre')
                    + total_direct_cost * self.ssc.get('csp.dtr.cost.epc.percent')/100
                    + net_capacity * 1e6 * self.ssc.get('csp.dtr.cost.epc.per_watt')
                    + self.ssc.get('csp.dtr.cost.epc.fixed') )
        
        sales_tax_cost = (total_direct_cost * self.ssc.get('csp.dtr.cost.sales_tax.percent')/100
                          * self.ssc.get('sales_tax_rate')/100)
        total_indirect_cost = land_cost + epc_cost 
        total_installed_cost = total_direct_cost + total_indirect_cost + sales_tax_cost
        return total_installed_cost

    @staticmethod
    def estimate_receiver_pumping_parasitic():
        """Estimates receiver pumping parasitic power for dispatch parameter

        .. note::
            This function assumes a constant value because troughs pressure drop is difficult to estimate reasonably

        Returns:
            Receiver pumping power per thermal rating [MWe/MWt]
        """
        return 0.0125  # [MWe/MWt]

    @staticmethod
    def get_plant_state_io_map() -> dict:
        io_map = {  # State:
                  # Number Inputs                         # Arrays Outputs
                  'defocus_initial':                      'defocus_final',
                  'rec_op_mode_initial':                  'rec_op_mode_final',
                  'T_in_loop_initial':                    'T_in_loop_final',
                  'T_out_loop_initial':                   'T_out_loop_final',
                  'T_out_scas_initial':                   'T_out_scas_last_final',        # array

                  'T_tank_cold_init':                     'T_tes_cold',
                  'T_tank_hot_init':                      'T_tes_hot',
                  'init_hot_htf_percent':                 'hot_tank_htf_percent_final',

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
        plant_state['init_hot_htf_percent'] = self.value('init_hot_htf_percent')
        plant_state.pop('T_out_scas_initial')   # initially not needed
        return plant_state

    def set_tes_soc(self, charge_percent):
        self.plant_state['init_hot_htf_percent'] = charge_percent

    @property
    def solar_multiple(self) -> float:
        return self.ssc.get('specified_solar_multiple')

    @solar_multiple.setter
    def solar_multiple(self, solar_multiple: float):
        """
        Set the solar multiple and updates the system model. Solar multiple is defined as the the ratio of receiver
        design thermal power over power cycle design thermal power.
        
        Args:
            solar_multiple: Solar multiple
        """
        self.ssc.set({'specified_solar_multiple': solar_multiple})

    @property
    def cycle_thermal_rating(self) -> float:
        return self.value('P_ref') / self.value('eta_ref')

    @property
    def field_thermal_rating(self) -> float:
        # TODO: This doesn't work with specified field area option
        return self.value('specified_solar_multiple') * self.cycle_thermal_rating

    @property
    def cycle_nominal_efficiency(self) -> float:
        return self.value('eta_ref')

    @property
    def number_of_reflector_units(self) -> float:
        """Returns number of solar collector assemblies (SCA) within the field."""
        return self.value('nSCA') * self.value('nLoops') * self.value('FieldConfig')

    @property
    def minimum_receiver_power_fraction(self) -> float:
        """Returns minimum field mass flowrate fraction."""
        return self.value('m_dot_htfmin')/self.value('m_dot_htfmax')

    @property
    def field_tracking_power(self) -> float:
        """Returns power load for field to track sun position in MWe"""
        return self.value('SCA_drives_elec') * self.number_of_reflector_units / 1e6  # W to MW

    @property
    def htf_cold_design_temperature(self) -> float:
        """Returns cold design temperature for HTF [C]"""
        return self.value('T_loop_in_des')

    @property
    def htf_hot_design_temperature(self) -> float:
        """Returns hot design temperature for HTF [C]"""
        return self.value('T_loop_out')

    @property
    def initial_tes_hot_mass_fraction(self) -> float:
        """Returns initial thermal energy storage fraction of mass in hot tank [-]"""
        return self.plant_state['init_hot_htf_percent'] / 100.

