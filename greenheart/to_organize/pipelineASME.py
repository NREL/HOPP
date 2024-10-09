import math
import numpy as np 
from matplotlib import pyplot as plt 
from greenheart.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals

verbose = False
class PipelineASME: 
    """    
    # ASME B31-12 Hydrogen Pipes, PL3.7.1 Steel Pipe design
    """
    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict 
        self.output_dict = output_dict
    
        #self.pipeline_model = input_dict['pipeline_model']
        self.pipe_diam_in = input_dict['pipe_diam_in']
        self.pipe_thic_in = input_dict['pipe_thic_in']
        self.dist_to_h2_load_km = input_dict['dist_to_h2_load_km']
        self.site_depth_m = input_dict['site_depth_m']
        self.h2_flow_kg_h = input_dict['flow_rate_kg_hr']
        self.pres_in_bar = input_dict['pressure_bar']
        self.steel_cost_ton = input_dict['steel_cost_ton']
        #self.offshore_param = input_dict['offshore_bool']
        #self.plant_life = input_dict['plant_life']
        #self.useful_life = input_dict['useful_life']

        # Assumptions
        self.useful_life = 30       #[years]
        self.plant_life = 30        # [years]

        # TODO: Calculate density as a function of temperature (ambient?)
        self.rho_h2 = 0.08375 # STP H2 density, kg/m**3
        self.mu_h2 = 0.88e-5 # Pa-s at 20C

        # Undersea environment 
        self.rho_h2o = 1026 # kg/m**3
        self.g = 9.81 # m/s**2 

        # Estimates from Weight/Price charts 
        # Example: https://www.savoypipinginc.com/blog/live-stock-and-current-price.html
        #self.Smys = 52000        # Min. Yield Hoop Stress (Table IX-1B)
        self.rho_steel =  7529.0 # kg/m**3 = 470.0 lb / ft**3 estimate 
        #self.cost_per_kg = 2.80 # $ per kg 
        self.Re = 2100.0
        
        # ASME B31.12 Hydrogen Pipeline parameters 
        self.F = 0.5             # Design factor (3.7.1.1)
        self.T = 1.0             # Temperature factor: < 250degF (3.7.1-3)
        self.Smin = 52000        # X52 Carbon Steel
        self.E = 1.0             # Long. Weld Factor: Seamless (Table IX-3B)
        self.Hf = 1.0            # Table IX-5b 
        self.pipe_sch = 40       # common pipe schedule for Hoop Stress estimation

        # Conversions factors
        self.in2cm = 2.54
        self.bar2psi = 14.5
        self.bar2Pa = 100000.0
        
    def pipelineDesign(self): 
        #flow_rate_kg_per_hr, inlet_pressure_bar, outlet_pressure_bar, length_m):
        """
        Calculate pressure drop based on General Flow Equation (Darcy eq.)
        Inputs
        ======
         gas flow rate (kg/hr)
         distance (km)
         depth (m)
         pipe diameter (in)
         pipe thickness (in)

         Output
         ======
         pressure Drop diameter (Pa)
         pipe thickness (in)
        """

        # Calculate pipeline distance to total length of pipe
        # TODO: Replace percent_added_length with bends, drops, connections
        site_depth_km = self.site_depth_m / 1000.0 # m to km
        percent_added_length = 0.02 # 2%

        total_length = (1+percent_added_length)* self.dist_to_h2_load_km + 2 * site_depth_km

        self.output_dict['total_pipeline_length_km'] = total_length

        # Volumetric Flow Rate
        flow_rate_kg_per_sec = self.h2_flow_kg_h / 3600.0   # kg/h to kg/s
        Q = flow_rate_kg_per_sec / self.rho_h2              # m**3/s
        if verbose: print("Vol. Flow rate (m**3/sec):", Q) 

        # Pipe Dimensions
        L_m = total_length * 1000.0                     # km to m
        d_m = self.pipe_diam_in * self.in2cm / 100.0    # inch to meters 
        A = (np.pi/4) * d_m**2                          # m**2

        # Velocity and Reynolds Number 
        V = Q / A                                   # m/s
        Re = (self.rho_h2 / self.mu_h2) * V * d_m
        if verbose: print("Reynolds number:", Re)
        
        # Friction factor (assuming laminar flow)
        # TODO: Find a more accurate number for relative roughness and better Reynolds estimate 
        # if(Re < 4001.0): 
        f = 64/Re       # laminar flow
        #else: 
        #   f = 0.03        # Approx. from Moody Diagram assume Re ~10^4 and relative roughness ~0.002 
        if verbose: print("Friction factor:", f)
    
        # Design pressure option - credit undersea pressure 
        P_hydro_Pa = self.rho_h2o * self.g * self.site_depth_m # N/m**2
        press_credit = False
        if press_credit:
            P_design_Pa = self.pres_in_bar * self.bar2Pa - P_hydro_Pa
        else:
            P_design_Pa = self.pres_in_bar *self.bar2Pa
        
        P_design_psi = P_design_Pa * (self.bar2psi / self.bar2Pa )
        self.output_dict['pres_design_bar'] = P_design_psi / self.bar2psi
        
        # Approximating The Minimum Yield Stress based on SCH40 pipe
        # TODO: Expand for either more SCH's
        SYMS = P_design_psi * 1000.0 / self.pipe_sch
        
        if SYMS <=52000: 
            self.Smin = 52000
        else:
            self.Smin = SYMS

        self.output_dict['syms_sch40pipe'] = self.Smin
        # Design pressure from Barlow's Formula
        S = self.Smin * self.F * self.E * self.T * self.Hf   # Design Hoop Stress

        # Create 2D array of pressures based on diameter and thickness for a given hoop stress
        P_calc_psi = np.zeros([np.size(self.pipe_diam_in),np.size(self.pipe_thic_in)])
        for i in range(np.size(self.pipe_diam_in)):
            for j in range(np.size(self.pipe_thic_in)): 
                P_calc_psi[i,j] = 2 * self.pipe_thic_in[j] * S / self.pipe_diam_in[i]

        if verbose: print("Pressure (bar):", P_calc_psi/self.bar2psi)
        self.output_dict['pres_calc_bar'] = P_calc_psi / self.bar2psi
        
        # Find the thickness and diameter that meet the design pressure 
        # TODO: Come up with a higher fidelity method to find values in a 2D array 
        tol = 0.01 * self.output_dict['pres_design_bar'] # Within a 10% tolerance 
        d_Ind, t_Ind = np.where(np.abs(self.output_dict['pres_calc_bar'] - self.output_dict['pres_design_bar']) < tol)
        # print(d_Ind)
        if verbose: print(self.pipe_diam_in[d_Ind])
        # print(t_Ind)
        if verbose: print(self.pipe_thic_in[t_Ind])

        self.output_dict['design_diam_in'] = self.pipe_diam_in[d_Ind]
        self.output_dict['design_thic_in'] = self.pipe_thic_in[t_Ind]

            # Pressure Drop from Darcy's Equation (General Flow Equation)
            # TODO: Find outlet pressure and pressure drop
            # delP = 8.0 * ( self.rho_h2 * f * L_m * Q**2 ) / ((self.pipe_diam_in[i] * np.pi**2)**(1/5))
            #self.output_dict['pressure_drop_Pa'] = delP 
            #self.output_dict['pressure_out'] = self.pres_design_bar - delP / 100000.0 # bars 

    def pipelineCost(self):
        # Designed diameter, thickness, and total length all in meters 
        d_m = self.output_dict['design_diam_in']  * self.in2cm / 100.0 
        t_m = self.output_dict['design_thic_in']  * self.in2cm / 100.0
        L_m = self.output_dict['total_pipeline_length_km'] * 1000.0

        # Find total surface area, volume, and mass of steel required for pipeline
        pipe_cost_total = np.zeros([np.size(d_m)])

        for i in range(np.size(d_m)):
            steel_volume = np.pi * d_m[i] * t_m[i] * L_m             # m**3 
            steel_mass = steel_volume * self.rho_steel     # kg

            pipe_cost_total[i] = steel_mass * self.steel_cost_ton / 1000.0   # USD
            # TODO: Figure out what PPI to $USD per weight is 
            # fab_cost = mat_cost * 256/100 # PPI for tube manufacturing https://fred.stlouisfed.org/series/PCU33121033121002 

        if verbose: print("Design pressure Pipe Cost:",pipe_cost_total) 
        
        # Find the cost of installation based on ORBIT's example S-lay vessel
        pipe_install_hrs = self.output_dict['total_pipeline_length_km'] / 0.15
        slay_vessel_daily_rate = 0.44e6
        pipe_install_cost = ( pipe_install_hrs / 24.0 ) * slay_vessel_daily_rate

        # Cost of the hybrid substation based on ORBIT's example 
        substation_design_cost = 20e6
        substation_mass = 21000 # tons of steel for install 
        substructure_mass = 0.4 * substation_mass
        pile_mass = 8 * substructure_mass ** 0.5574
        total_substructure_mass = substructure_mass + pile_mass
        struc_steel_cost = 3000.0 # $USD/ton
        fabrication_rate = 14500 # $USD/ton
        substructure_cost = total_substructure_mass * struc_steel_cost + substation_mass * fabrication_rate
        substation_install_days = 14
        subs_vessel_daily_rate = 0.5e6
        substation_install_cost = substation_install_days * subs_vessel_daily_rate 

        # Total CapEx
        capex_pipe = pipe_cost_total + pipe_install_cost
        capex_substation = substation_design_cost + substructure_cost + substation_install_cost

        # Find the cost of construction and loading/unloading 
        self.output_dict['pipeline_capex'] = capex_pipe
        self.output_dict['substation_capex'] = capex_substation
        
        # Opex Factor based on on-land or undersea (7%) 
        # https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub 
        opex_pipe = 7/100 * capex_pipe / self.plant_life    # $US/year

        self.output_dict['pipeline_opex'] = opex_pipe

# Test sections 
if __name__ == '__main__':
    print("PipelineASME Testing section")
    in_dict = dict()
    in_dict['steel_cost_ton'] = 900.0 # $ US/ton searching for seamless FBE X52 carbon steel > $500-$1000 per ton

    in_dict['dist_to_h2_load_km'] = 80.0 
    in_dict['site_depth_m'] = 45.0
    in_dict['pipe_diam_in'] = np.linspace(12.0, 48.0, 20)
    in_dict['pipe_thic_in'] = np.linspace(0.1, 2.0, 50)

    in_dict['flow_rate_kg_hr'] = 125
    in_dict['pressure_bar'] = 100.0 
    
    out_dict = dict()

    pipeline_test = PipelineASME(in_dict,out_dict)
    pipeline_test.pipelineDesign()
    pipeline_test.pipelineCost()
 
    #print("Min. Hoop Stress SCH40 pipe:", out_dict['syms_sch40pipe'])
    print("Pipeline Length (km):", out_dict['total_pipeline_length_km'])
    print("Pipeline Design Pressure (bar):",in_dict['pressure_bar'])
    print("Pipeline Diameter: {} in, Thickness {} in".format(out_dict['design_diam_in'][0],out_dict['design_thic_in'][0]))
    print("Pipeline Cost ($US/km): ", np.min(out_dict['pipeline_capex']/out_dict['total_pipeline_length_km']))
    print("Substation Cost: ", out_dict['substation_capex'])
    print("Total H2-Export CapEx:", out_dict['substation_capex']+np.min(out_dict['pipeline_capex']))
    print("Pipeline Opex ($US/year)", np.min(out_dict['pipeline_opex']))
    
    # Contour Plot
    #fig, ax = plt.subplots()
    fig, axs = plt.subplots(figsize=(6, 4))
    #fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    #CS1 = axs.contour(in_dict['pipe_diam_in'],in_dict['pipe_thic_in'], \
    #                  np.transpose(out_dict['pres_calc_bar']), levels = [out_dict['pres_design_bar'] ], \
    #                   colors=('k',),linestyles=('--',),linewidths=(3,))
    CS = axs.contour(in_dict['pipe_diam_in'], in_dict['pipe_thic_in'], np.transpose(out_dict['pres_calc_bar']))
    CS1 = axs.contour(in_dict['pipe_diam_in'], in_dict['pipe_thic_in'], np.transpose(out_dict['pres_calc_bar']), \
                      levels=[out_dict['pres_design_bar']], colors=('k'), linestyles=('--'),linewidths=(2))

    # def fmt(x):
    #     s = f"{x:.1f}"
    #     if s.endswith("0"):
    #         s = f"{x:.0f}"
    #     return rf"{s} " if plt.rcParams["text.usetex"] else f"{s} "
    axs.clabel(CS, CS.levels, inline=True, fontsize=10)
    axs.clabel(CS1, CS1.levels, inline=True, fontsize=10)

    axs.set_title("Pipeline Pressure Analysis")
    axs.set_xlabel("Pipe Diameter (in)")
    axs.set_ylabel("Pipe Thickness (in)")

    plt.figure(figsize=(6,4))
    plt.plot(out_dict['design_diam_in'],out_dict['pipeline_capex']/out_dict['total_pipeline_length_km'])
    plt.xlabel('Pipe Diameter (in)')
    plt.ylabel('Unit Cost ($US/km)')
    plt.title("Cost of Pipeline \n Design Pressure:{}bar".format(out_dict['pres_design_bar']))
    plt.show()
    