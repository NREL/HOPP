# from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
# from numpy.lib.function_base import average
# import examples.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd
from power_electronic_info import *
#from HOPP.tools.resource import resource
#from hOPP.hybrid.add_custom_modules.custom_wind_floris import custom_wind_floris
#from hybrid.add_custom_modules.custom_wind_floris import Floris
#from hybrid.pv_source import PVPlant

resource_filepath = 'HOPP/resource_files/'
class set_pv_PE_component_sizing:
      def __init__(self,pv_rated_power_kWdc,load_connection_type):
            self.pv=solar_panel_PE_config()
            self.pv_cables=ac_cables()
            dc2ac_loading_ratio = 1.3 #used for AC and DC coupled systems
            #step1) size inverter
            #step2) find cable/layout
            #step3) size load connection component
            pass
      def pv_simple_layout(self,pv_rated_power_kWdc):
            #inverter per string
            #Sunny Tripower Core1 33-US/50-US/62-US inverter specs from solaris
            n_panels=np.ceil(pv_rated_power_kWdc*1000/self.pv.Pmax_Wdc_ref)
            self.pv.ref_panel_width_m
            self.pv.ref_panel_length_m
            nmax_panel_series = np.floor(self.pv_cables.Vline/self.pv.V_oc_ref)

            self.pv_cables.Vline
            self.pv.I_sc_ref #use to determine layout
            #current same in series
            #voltage same in parallel
            #Vdc=3*sqrt(2)*Vac_l2l/pi = 1.35 Vac_l2l

      def pv_to_collection_system(self):
            dc2ac_inverter(size)
            ac_cables
      
      def pv_collection_system_to_dc_load(self):
            dc2ac_inverter(size)
            
            pass
      def pv_collection_system_to_ac_load(self):
            ac2ac_transformer(size)
            pass
      def central_pv_to_ac_load(self):
            dc2ac_inverter(size)
            pass
      def central_pv_to_dc_load(self):
            dc2dc_bidirect_converter(size)
            pass

      # def ac_connected_central_pv(self):
      #       pass
      # def dc_connected_central_pv(self):
      #       pass
      # def dc_connected_distributed_pv(self):
      #       ac_cables
      #       pass

class set_wind_PE_component_sizing:
      def __init__(self,nTurbs,layout_x,layout_y,turbine_power_rating_kW):
            self.turbine=turbine_PE_config(nTurbs,layout_x,layout_y,turbine_power_rating_kW)
            self.turbine_cables=ac_cables()
            #self.turbine_cables.Vline
            #self.turbine.Vline = 480
            
      def size_transformer(self):
            #size transformer
            self.ac2ac=ac2ac_transformer(size)
            nTurns=self.ac2ac.calc_nTurns_from_voltage(self.turbine.Vline,self.turbine_cables.Vline)
            I_out_rated=self.ac2ac.calc_output_current(self.turbine.Iline_rated,nTurns)
            
      def choose_cable(self):
            pass
      def turbine_layout_for_cabling(self,layout_x,layout_y):


      #self.turbine_cables.find_possible_cables(max_power_to_cable_kWac,max_current_to_cable)
            pass
class set_battery_PE_component_sizing:
      def __init__(self,nBattery,charge_rate_kW_per_battery,capacity_kWh_per_battery):
            self.battery=battery_PE_config()
            pe_size=charge_rate_kW_per_battery
            inv2bat_storage_ratio = 1.67 #only used for AC coupled-systems

            dc2dc_bidirect_converter(size)
            pass


class set_load_PE_component_sizing:
      def __init__(self,load_rated_power,grid_connected):
            self.grid=grid_connection_PE_config
            #TODO: make below inputs!
            load_rating_kW=1000
            load_Vtype='DC'
            load_Imax=2000
            load_Vmax=1000
            load_PE_config(load_rating_kW,load_Vtype,load_Imax,load_Vmax)
            pass
      def power_to_grid(self):
            self.grid=grid_connection_PE_config()
            ac2ac_transformer(size) #step-up
            pass
      def power_from_grid(self):
            pass
      def ac_central_load(self):
            #ac collection system to AC load
            ac2ac_transformer(size) #?
      def dc_central_load (self):
            #AC collection system to DC load
            ac2dc_unidirect_rectifier(size)
            #maybe
            dc2dc_bidirect_converter(size)
            
class all_sys_setup:
      def __init__(self,load_types,generation_components,connection_types,grid_connection_scenario):
            
            
            '''Here is where the user will input everything they want
            I.e., load types would be grid and/or AC load or DC load and the
            relevant information
            generation_components are just what is included and their sizing
            and general layout
            connection_types is what generation component is connected to what load
            and how
            ex: distributed pv to central battery
            ex: central pv to central battery
            ex: distributed pv to distributed battery to central loc
            ex: distributed pv to central AC or DC load
            ex: battery to AC or DC load
            ex: battery to grid
            ex: wind to central 
            '''
            if grid_connection_scenario=='off-grid':
                  pass
            elif grid_connection_scenario=='hybrid-grid':
                  #is grid a source or a load?!
                  pass
            elif grid_connection_scenario=='grid-only':
                  pass
            