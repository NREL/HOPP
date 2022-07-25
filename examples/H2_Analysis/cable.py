## Basic Cable Model referenced from NREL/WISDEM/ORBIT tool 
'''
    Cost relationship for HVDC and substation gathered from ORBIT (offshore BOS tool) 
    by varying distance as an input from 40km-150km. These numbers are using the 
    Gulf of Mexico site (@45m depth). 
        Spot checking indicates that cost scales linearly with distance 

    "Array" refers to the MVAC cables between turbines to substation
    "Export" refers to the HVDC cables between substation and shore

    Currently the electrical-refactor branch contains HVDC for export only. 
    Discussions with Jacob Nunemaker and Patrick Duffy clarified input details of ORBIT. 

    Input
    =====
        Distance between shore and substation 

    Output
    ======
        Array Cable System Cost 
        Export Cable System Cost 
'''
import math
import numpy as np 
from matplotlib import pyplot as plt 
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals

# Export Cable Model - HVDC  
def exportCable(dist_to_shore_km):
    # Transmission losses  
    #loss = (3.5 / 100) * self.dist_to_shore_km / 1000.0 
    #efficiency = 1 - loss  
    
    system_cost = 1.0041 * dist_to_shore_km + 10.07 
    install_cost = 0.0517 * dist_to_shore_km + 173.86
    
    cableCapEx = system_cost + install_cost
    cableOpEx = 0.5/100 * cableCapEx 

    return cableCapEx, cableOpEx

# Export Substation Model - HVDC
def exportSubstation(dist_to_shore_km): 

    system_cost = 233.76     
    install_cost = 0.0129 * dist_to_shore_km + 2.9527
     
    stationCapEx = system_cost + install_cost 
    
    return stationCapEx

#TODO: include OpEx factor based on CapEx from
# OpEx was referenced in the following report to be 0.5% for offshore HVDC 
# https://www.sciencedirect.com/science/article/pii/S0360319921009137?via%3Dihub 

#TODO: Potentially develop a wrapper to call ORBIT to calculate BOS elements. 
# - need to figure out how to create a .yaml scenario file based on  
    
# Test sections 
if __name__ == '__main__': 
    print('### cable test ###')
    in_dict = dict()
    in_dict['dist_to_shore_km'] = 80.0
    
    out_dict = dict()

    cabletest = Cable(in_dict, out_dict)
    cabletest.transmissionLoss()
    cabletest.arrayCables()
    cabletest.exportCables()
    cabletest.subStation()

    print("MVAC Array System Cost (mUSD):", out_dict['array_cost'])
    print("HVDC Export System Cost (mUSD): ", out_dict['export_cost'])
    print("StationCost (mUSD):", out_dict['station_cost'])
    print("Tranmission Efficiency (%):", out_dict['trans_eff']*100)
     
#OLD
    #def arrayCables(self):
    #    systemCost = 66.51
    #    installCost = 0.0049 * self.dist_to_shore_km + 31.306

    #    self.output_dict['array_cost'] = systemCost + installCost 
