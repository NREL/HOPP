# from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
# from numpy.lib.function_base import average
# import examples.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
#from HOPP.tools.resource import resource
#from hOPP.hybrid.add_custom_modules.custom_wind_floris import custom_wind_floris
#from hybrid.add_custom_modules.custom_wind_floris import Floris
#from hybrid.pv_source import PVPlant
#https://www.sciencedirect.com/science/article/pii/S0306261916312879
class solar_panel_PE_config:
      def __init__(self):#,nPanels,panel_rating_kWdc,layout_x,layout_y):
            ##cited by ATB [2021]: https://www.nrel.gov/docs/fy22osti/80694.pdf
            self.V_type = 'DC'
            # pv_Vac_l2l= 480
            # I_inv_output = 64 #[A]
            # Peimar 400 watt mono XL solar Panel SM400M
            self.Pmax_Wdc_ref = 400
            self.V_mpp_ref = 41.3 
            self.I_mpp_ref = 9.69
            self.V_oc_ref = 50.39 
            self.I_sc_ref=10.26
            self.ref_panel_width_m=1.98
            self.ref_panel_length_m=1
            #max_fuse_series_I = 15
            #https://github.com/NREL/ssc/blob/develop/ssc/cmod_pvwattsv8.cpp
      def ref_pv_configs_series_connected(self,panel_rating_kWdc):
            #Peimar_EN_SM400M 
            # Peimar 400 watt mono XL solar Panel SM400M
            # Pmax_Wdc_ref = 400 #max power a asolar panel can produce
            # V_mpp_ref = 41.3 #max voltage to get Pmax
            # I_mpp_ref = 9.69 #max current to get Pmax
            # V_oc_ref = 50.39 #use to determine
            # #V_oc: number to use when determining how many solar panels
            # #to wire in series, max voltage solar panel can produce
            # #when not connected to a load (open-circuit voltage)
            # I_sc_ref=10.26 # short circuit current
            # #used to size amperage of connected devices
            # max_fuse_series_I = 15
            n_panels_in_series = panel_rating_kWdc*1000/self.Pmax_Wdc_ref
            V_oc=self.V_oc_ref*n_panels_in_series
            V_mpp=self.V_mpp_ref*n_panels_in_series
            I_mpp=self.I_mpp_ref
            I_sc=self.I_sc_ref
            
            return V_mpp,I_mpp,V_oc,I_sc

class ac_cables:
      def __init__(self):
            self.V_type = '3-phase AC'
            self.n_max=10
            self.Vline=34.5*1000

            kiloft_to_km = 0.3048
            cable_resistance_per_kft = np.array([0.12,0.25,0.02,0.01,0.009])
            self.cable_names = np.array(["AWG 1/0","AWG 4/0","MCM 500","MCM 1000","MCM 1250"])
            self.cable_resistance_per_m = (cable_resistance_per_kft *(1/kiloft_to_km))/(1e3)
            self.cable_ampacity = np.array([150,230,320,455,495])
            self.cable_cost_per_m = np.array([61115.1602528554,72334.3683802817,96358.26769213431,104330.7086713996,115964.28690974298])/1000

      def find_possible_cables(self,max_power_to_cable_kWac,max_current_to_cable):
            #max_current_to_cable = i_turb*n_turbs
            n_cables = np.ceil(max_current_to_cable/self.cable_ampacity)
            p_line_max = np.sqrt(3)*max_current_to_cable*n_cables*self.Vline
            cb_idx = np.argwhere((p_line_max >= max_power_to_cable_kWac) & (n_cables<=self.n_max))
            cb_idx = cb_idx.reshape(len(cb_idx))

            i_per_cable = max_current_to_cable/n_cables[cb_idx]
            cable_r = self.cable_resistance_per_m[cb_idx]
            i_rated_cable = self.cable_ampacity[cb_idx]
            n_cables_okay = n_cables[cb_idx]
            names = self.cable_names[cb_idx]
            
            p_loss_per_cable = (i_per_cable**2)*cable_r
            p_loss_tot_per_m = p_loss_per_cable*n_cables[cb_idx]
            idx_lowestloss = np.argmin(p_loss_tot_per_m)
            possible_cable_configs=pd.DataFrame({'Cable Type':names,'Rated Ampacity':i_rated_cable,
            'Cable Resistance/meter':cable_r,'n_cables':n_cables_okay,'Current per Cable':i_per_cable,
            'Power Loss per Cable':p_loss_per_cable,'Total Power Loss per m':p_loss_tot_per_m})


            return possible_cable_configs

class turbine_PE_config:
      def __init__(self,nTurbs,layout_x,layout_y,turbine_power_rating_kW):
            #assuming constant voltage
            self.V_type = '3-phase AC'
            self.Vline = 480
            self.P_rated_kW = turbine_power_rating_kW
            self.n_components=nTurbs
            self.Iline_rated=(turbine_power_rating_kW*1e3)/(np.sqrt(3)*self.Vline)
            dist_between_turbs = np.max(np.diff(layout_x))
            x_coord,cnt_x=np.unique(layout_x,return_counts=True)
            y_coord,cnt_y=np.unique(layout_y,return_counts=True)
      def calc_turbine_current(self,turbine_power_kW):
            I = turbine_power_kW*1e3/(np.sqrt(3)*self.Vline) #[Amps]
            return I

class grid_connection_PE_config:
      def __init__(self,pv_inputs):
            #power station has low-voltage (2.3 to 30 kV)
            #stepped-up by power-station transformer
            #to 115 kV -765 kV AC
            self.V_in = 115*1000
            self.V_type = '3-phase AC'
            self.is_load = True
            pass
class load_PE_config:
      def __init__(self,load_rating_kW,load_Vtype,load_Imax,load_Vmax):
            self.Vin_type=load_Vtype
            self.P_rated_kW = load_rating_kW
            self.Imax = load_Imax
            self.Vmax = load_Vmax
class battery_PE_config:
      def __init__(self):
            #ATB: https://atb.nrel.gov/electricity/2022/utility-scale_battery_storage
            #cited by ATB [2021]: https://www.nrel.gov/docs/fy22osti/80694.pdf
            self.V_type = 'DC'
            self.charge_battery = 'DC/DC'
            self.discharge_battery = 'DC/DC'
            pass
class ac2dc_unidirect_rectifier:
      def __init__(self,size_kWac):
            self.Vin_type = '3-phase AC'
            self.Vout_type = 'DC'
            self.rated_input_power = size_kWac
            ##rectification mode: forward power flow (AC-> DC)
            x_load_percent = np.linspace(0.0,1.0,11)
            self.ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
            self.dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
            ac2dc_eff = self.ac2dc_rectification_eff*self.dc2dc_rectification_eff
            self.f = interpolate.interp1d(x_load_percent,ac2dc_eff)
            pass

class ac2ac_transformer:
      def __init__(self,size_kWac):
            # efficiency curves: https://www.mdpi.com/2076-3417/9/3/582
            self.rated_input_power = size_kWac
            self.Vin_type = '3-phase AC'
            self.Vout_type = '3-phase AC'
            x_load_percent = np.linspace(0.0,1.0,11)
            ac2ac_transformer_eff=np.array([90,90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])/100
            self.f = interpolate.interp1d(x_load_percent,ac2ac_transformer_eff)

      def calc_nTurns_from_voltage(self,rated_input_voltage,rated_output_voltage):
            nTurns = rated_output_voltage/rated_input_voltage
            return nTurns
      def calc_nTurns_from_current(self,rated_input_current,rated_output_current):
            nTurns = rated_input_current/rated_output_current
            #double check!
            return nTurns
      def calc_output_current(self,input_current,nTurns):
            I_out = input_current/nTurns
            return I_out
      def calc_output_voltage(self,input_voltage,nTurns):
            V_out = input_voltage*nTurns
            return V_out
class dc2dc_unidirect_converter:
      def __init__(self,size_kWdc):
            self.rated_input_power = size_kWdc
            self.Vin_type = 'DC'
            self.Vout_type = 'DC'
            ## efficiency curves: https://www.mdpi.com/2076-3417/9/3/582
            x_load_percent=np.array([0,10.1, 13.2,  17.1,  20. ,  25.1,  29.9,  35. ,  40.1,  50.1,  60. , 69.9,  79.9,  90.1, 100. ])/100
            dc2dc_transformer_eff=np.array([80,80.9,84.1, 88.1, 90. , 93.2, 95. , 96. , 96.3, 96.2, 95.9, 95.4, 95.3, 95.3, 95.2])/100
            f_dc2dc=interpolate.interp1d(x_load_percent,dc2dc_transformer_eff)
            self.f = interpolate.interp1d(x_load_percent,dc2dc_transformer_eff)
      def calc_nTurns_from_voltage(self,rated_input_voltage,rated_output_voltage):
            nTurns = rated_output_voltage/rated_input_voltage
            return nTurns
      def calc_nTurns_from_current(self,rated_input_current,rated_output_current):
            nTurns = rated_input_current/rated_output_current
            return nTurns
      def calc_output_current(self,input_current,nTurns):
            I_out = input_current/nTurns
            return I_out
      def calc_output_voltage(self,input_voltage,nTurns):
            V_out = input_voltage*nTurns
            return V_out
class dc2dc_bidirect_converter:
      def __init__(self,size_kWdc):
            ##https://www.sciencedirect.com/science/article/pii/S0306261916312879
            self.rated_input_power = size_kWdc
            self.Vin_type = 'DC'
            self.Vout_type = 'DC'
            x_load_percent = np.linspace(0.0,1.0,11)
            dc2dc_bidirectconverter_eff=np.array([91,91.44, 95.19, 96.56, 97.18, 97.50, 97.62, 97.68, 97.77, 97.68, 97.68])/100
            f_dc2dc=interpolate.interp1d(x_load_percent,dc2dc_bidirectconverter_eff)
            self.f = interpolate.interp1d(x_load_percent,dc2dc_bidirectconverter_eff)
# efficiency curves: https://www.mdpi.com/2076-3417/9/3/582
class dc2ac_inverter:
      def __init__(self,size_kWdc):
            #[Kim,2013] Figure 20: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6269110&tag=1
            self.rated_input_power = size_kWdc
            self.Vin_type = 'DC'
            self.Vout_type = '3-phase AC'
            x_load_percent = np.linspace(0.0,1.0,11)
            ac2dc_generation_eff=np.array([96,96.53, 98.00, 98.12, 98.29, 98.03, 97.91, 97.74, 97.15, 96.97, 96.48])/100
            dc2dc_generation_eff=np.array([90,91.44, 95.19, 96.56, 97.18, 97.50, 97.62, 97.68, 97.77, 97.68, 97.68])/100
            inverter_eff = ac2dc_generation_eff*dc2dc_generation_eff
            self.f = interpolate.interp1d(x_load_percent,inverter_eff)
# resource_filepath = 'HOPP/resource_files/'
# x_load_percent = np.linspace(0.0,1.0,11)

# ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
# dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100


# ac2ac_transformer_eff=np.array([90,90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])/100
# ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
# dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
# rect_eff = ac2dc_rectification_eff*dc2dc_rectification_eff


# class ac2dc_bidirect_rectifier:
#       def __init__(self,pv_inputs):
#             #[Kim,2013] Figure 20: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6269110&tag=1
#             x_load_percent = np.linspace(0.0,1.0,11)
#             #rectification mode: forward power flow (AC-> DC)
#             #bi-directionary ac/dc converter
#             self.ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
#             self.dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
#             ac2dc_eff = self.ac2dc_rectification_eff*self.dc2dc_rectification_eff
#             self.f_ac2dc = interpolate.interp1d(x_load_percent,ac2dc_eff)
#             #generation mode: backward power (DC->AC)
#             self.ac2dc_generation_eff=np.array([96,96.53, 98.00, 98.12, 98.29, 98.03, 97.91, 97.74, 97.15, 96.97, 96.48])/100
#             self.dc2dc_generation_eff=np.array([91,91.44, 95.19, 96.56, 97.18, 97.50, 97.62, 97.68, 97.77, 97.68, 97.68])/100
#             dc2ac_eff = self.ac2dc_generation_eff*self.dc2dc_generation_eff
#             self.f_dc2ac = interpolate.interp1d(x_load_percent,dc2ac_eff)
#             pass