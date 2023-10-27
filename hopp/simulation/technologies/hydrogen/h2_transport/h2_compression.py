'''
    Author: Jamie Kee
    Feb 7, 2023
    Source for most equations is HDSAM3.1, H2 Compressor sheet
    Output is in 2016 USD
'''

from numpy import interp, mean, dot
from math import log10, ceil, log

class Compressor:
    def __init__(self, p_outlet, flow_rate_kg_d, p_inlet=20, n_compressors=2, sizing_safety_factor=1.1):
        '''
            Parameters:
            ---------------
            p_outlet: oulet pressure (bar)
            flow_Rate_kg_d: mass flow rate in kg/day
        '''
        self.p_inlet = p_inlet # bar
        self.p_outlet = p_outlet # bar
        self.flow_rate_kg_d = flow_rate_kg_d # kg/day

        self.n_compressors = n_compressors # At least 2 compressors are recommended for operation at any given time
        self.n_comp_back_up = 1 # Often times, an extra compressor is purchased and installed so that the system can operate at a higher availability.
        self.sizing_safety_factor = sizing_safety_factor # typically oversized. Default to oversize by 10%

        if flow_rate_kg_d*(1/24)*(1/60**2)/n_compressors > 5.4:
            # largest compressors can only do up to about 5.4 kg/s
            """
            H2A Hydrogen Delivery Infrastructure Analysis Models and Conventional Pathway Options Analysis Results
            DE-FG36-05GO15032
            Interim Report
            Nexant, Inc., Air Liquide, Argonne National Laboratory, Chevron Technology Venture, Gas Technology Institute, National Renewable Energy Laboratory, Pacific Northwest National Laboratory, and TIAX LLC
            May 2008
            """
            raise ValueError("Invalid compressor design. Flow rate must be less than 5.4 kg/s per compressor")
            

    def compressor_power(self):
        R = 8.314 #J/mol-K
        T = 25+273.15 #K
        
        cpcv =  1.41 #H2 Cp/Cv ratio
        sizing = self.sizing_safety_factor # 110% based on typical industrial practices
        isentropic_efficiency = 0.88 # 0.88 based on engineering estimation for a reciprocating compressor

        # https://h2tools.org/hyarc/hydrogen-data/hydrogen-compressibility-different-temperatures-and-pressures
        Z_pressures = [1,10,50,100,300,500,1000] # Pressure for z correlation in bar
        Z_z = [1.0006,1.0059,1.0297,1.0601,1.1879,1.3197,1.6454] # H2 Compressibility aty 25C 
        Z = mean(interp([self.p_inlet,self.p_outlet],Z_pressures,Z_z))

        c_ratio_per_stage = 2.1 #based on engineering estimation for a reciprocating compressor operating on hydrogen
        self.stages = ceil((log10(self.p_outlet)-log10(self.p_inlet))/log10(c_ratio_per_stage))
        flow_per_compressor = self.flow_rate_kg_d/self.n_compressors #kg/d
        flow_per_compressor_kg_mols_sec= flow_per_compressor/24/60/60/2.0158 #convert units for power equation
        p_ratio = self.p_outlet/self.p_inlet
        theorhetical_power = Z*flow_per_compressor_kg_mols_sec*R*T*self.stages*(cpcv/(cpcv-1))*((p_ratio)**((cpcv-1)/(self.stages*cpcv))-1) #kW per compressor assumes equal work by all stages and intercooling to oringal temp
        actual_power = theorhetical_power/isentropic_efficiency #kW per compressor
        motor_efficiency = dot([0.00008,-0.0015,0.0061,0.0311,0.7617],[log(actual_power)**x for x in [4,3,2,1,0]])
        self.motor_rating = sizing*actual_power/motor_efficiency #kW per unit
    
    def compressor_system_power(self):
        return self.motor_rating, self.motor_rating*self.n_compressors # [kW] total system power

    def compressor_costs(self):
        n_comp_total = self.n_compressors + self.n_comp_back_up # 2 compressors + 1 backup for reliability
        production_volume_factor = 0.55 # Assume high production volume
        CEPCI = 1.29/1.1 #Convert from 2007 to 2016$

        cost_per_unit = 1962.2*self.motor_rating**0.8225*production_volume_factor*CEPCI
        if self.stages>2:
            cost_per_unit = cost_per_unit * (1+0.2*(self.stages-2))

        install_cost_factor = 2

        direct_capex = cost_per_unit*n_comp_total*install_cost_factor

        land_required = 10000 #m^2 This doesn't change at all in HDSAM...?
        land_cost = 12.35 #$/m2
        land = land_required*land_cost

        other_capital_pct = [0.05,0.1,0.1,0,0.03,0.12] # These are all percentages of direct capex (site,E&D,contingency,licensing,permitting,owners cost)
        other_capital = dot(other_capital_pct,[direct_capex]*len(other_capital_pct)) + land
        
        total_capex = direct_capex + other_capital

        ##
        # O&M
        ##
        CEPCI_labor = 1.15460860725178/1.08789393425957 #2013 to 2016
        labor_cost = 31.6*CEPCI_labor # Bureau of Labor Statistics- NAICS code# 2212 -- 2016$/man-hr
        labor_required = 288*(self.flow_rate_kg_d/100000)**0.25 #hrs/yr #Base case is 3 days per month for a 100,000 MJ/day useable capacity facility. Scaling factor of 0.25 is used for other sized facilities
        labor = labor_required*labor_cost

        electricity_consumption = self.motor_rating*8760*self.n_compressors
        electricity_price = 0.075 #  Maybe need to edit this?
        electricity = electricity_consumption*electricity_price

        #   All are percents of total capital investments, except for maint and repair is % of direct capex, overhead is % of labor
        other_fixed_pct = [0.01,0.01,0.001,0.04,0.5] # These are percentages of total capital investment - [insurance,property taxes,lic perm, maint repairs, overhead]
        other_fixed = dot(other_fixed_pct,[total_capex]*(len(other_fixed_pct)-2)+[direct_capex,labor])

        total_OM = labor+electricity+other_fixed

        return total_capex, total_OM
        
if __name__ == "__main__":
    p_inlet = 20 # bar
    p_outlet = 68 # bar
    flow_rate_kg_d = 9311
    n_compressors = 2

    comp = Compressor(p_outlet,flow_rate_kg_d, p_inlet=p_inlet, n_compressors=n_compressors)
    comp.compressor_power()
    motor_rating, power = comp.compressor_system_power()
    total_capex,total_OM = comp.compressor_costs() #2016$ , 2016$/y
    print("Power (kW): ", power)
    print(f'CAPEX: {round(total_capex,2)} $')
    print(f'Annual operating expense: {round(total_OM,2)} $/yr')

    # CAPEX: 680590.34 $
    # Annual operating expense: 200014.0 $/yr