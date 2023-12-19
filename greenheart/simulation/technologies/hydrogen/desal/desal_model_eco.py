################## needed addition ######################
"""
Description: This file already contains a desal model, but we need an estimate of the desal unit size, particularly mass and footprint (m^2)
Sources:
    - [1] Singlitico 2021 (use this as a jumping off point, I think there may be other good sources available)
    - [2] See sources in existing model below and the model itself
Args:
    - electrolyzer_rating (float): electrolyzer rating in MW
    - input and output values from RO_desal() below
    - others may be added as needed
Returns (can be from separate functions and/or methods as it makes sense):
    - mass (float): approximate mass of the desalination system (kg or tonnes)
    - footprint (float): approximate area required for the desalination system (m^2)
"""


#################### existing model ########################

## High-Pressure Reverse Osmosis Desalination Model
"""
Python model of High-Pressure Reverse Osmosis Desalination (HPRO).

Reverse Osmosis (RO) is a membrane separation process. No heating or phase change is necessary.
The majority of energy required is for pressurizing the feed water.

A typical RO system is made up of the following basic components:
Pre-treatment: Removes suspended solids and microorganisms through sterilization, fine filtration and adding chemicals to inhibit precipitation.
High-pressure pump: Supplies the pressure needed to enable the water to pass through the membrane (pressure ranges from 54 to 80 bar for seawater).
Membrane Modules: Membrane assembly consists of a pressure vessel and the membrane. Either sprial wound membranes or hollow fiber membranes are used.
Post-treatment: Consists of sterilization, stabilization, mineral enrichment and pH adjustment of product water.
Energy recovery system: A system where a portion of the pressure energy of the brine is recovered.

Costs are in 2013 dollars
"""
import sys
import numpy as np
from greenheart.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals



def RO_desal_eco(freshwater_kg_per_hr, salinity):
    """
    param: freshwater_kg_per_hr: Maximum freshwater requirements of system [kg/hr]

    param: salinity: (str) "Seawater" >18,000 ppm or "Brackish" <18,000 ppm
        
    output: feedwater_m3_per_hr: feedwater flow rate [m^3/hr] 
    
    output: desal_power: reqired power [kW] 

    output: desal_capex: Capital cost [USD]

    output: desal_opex: OPEX (USD/yr)

    Costs from: https://pdf.sciencedirectassets.com/271370/1-s2.0-S0011916408X00074/1-s2.0-S0011916408002683/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEcaCXVzLWVhc3QtMSJGMEQCIBNfL%2Frp%2BWpMGUW7rWBm3dkXztvOFIdswOdqI23VkBTGAiALG4NJuAiUkzKnukw233sXHF1OFBPnogJP1ZkboPkaiSrVBAjA%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMWZ%2Fh3cDnrPjUJMleKqkELlVPKjinHYk85KwguMS3panLr1RRD9qkoxIASocYCbkvKLE9xW%2BT8QMCtEaH3Is7NRZ2Efc6YFQiO0DHbRzXYTfgz6Er5qqvSAFTrfgp%2B5bB3NYvtDI3kEGH%2F%2BOrEiL8iDK9TmgUjojvnKt86zidswBSDWrzclxcLrw6dfsqZf6dVjJT2g3Cyy8LKnP9vc33tCbACRLeszW1Zce%2BTlBbON22W%2FJq0qLcXDxI9JpRDqL8T%2Fo7SsetEif2DWovTLnv%2B%2FX2tJotFp630ZTVpd37ukGtanjAr5pl0nHgjnUtOJVtNksHQwc8XElFpBGKEXmvRo2uZJFd%2BeNtPEB1dWIZlZul6B8%2BJ7D%2FSPJsclPfpkMU92YUStQpw4Mc%2FOJFCILFyb4416DsL6PVWsdcYu9bbry8c0hQGZlE7oXTFoUy9SKdpEOguXAUi3X4JxjZisy3esVH8zNS3%2FiFsNr2FkTB6MLaSjSKj344AuDCkQYZ7CnenAiCHgf4a2tSnfiXzAvAFnpeQkr4iCnZOQ4Eis6L3fVRpWlluX5HUpbvUMN6rvtmAzq0APJn1b3NmFHy4ORoemTGvmI%2FHTRYKuAu257XBMe7X1qAJlnmpt6yGXrelXCz%2FmUvmbT1SzxETA5ss4KR0OM4YdXNnFLUrsV44ZkUM%2B8FlwZr%2F%2FePjz4QeG4ApR821IYTyre3%2FY%2BBZxaMs5AcXKiTHGwfE7CDi%2BQQ7CnDKk0lleZcas6kxzDl9%2BmeBjqqAeZhBVwd5sEx6aDGxAQC0eWpux6HauoVfuPOCkkv621szF0kTBqcoOlJlJav4eUPW4efAzBremirjiRLI2GdP72lVqXz9oaCg5NFXeKJAWbWkLdzHnDOu8ecSUPn%2F0jcR2IO2mznLspx6wKQA%2BAPEVGgptkwZtDqHcw8FNx7Q8tWJ1C4qL1bEMl0%2FatDXOHiJfuzCFp4%2B4uijTNfpVXO%2BzYQuNJA7ZNUMroa&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230201T155950Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7RLVF2MG%2F20230201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a3770ee910f7f78c94bb84206538810ca03f7a653183191b3794c633b9e3f08f&hash=2e8904ff0d2a6ef567a5894d5bb773524bf1a90bc3ed88d8592e3f9d4cc3c531&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0011916408002683&tid=spdf-27339dc5-0d03-4078-a244-c049a9bb014d&sid=50eb5802654ba84dc80a5675e9bbf644ed4dgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f1650585c05065559515c&rr=792be5868a1a8698&cc=us
    
    A desal system capacity is given as desired freshwater flow rate [m^3/hr]
    """

    freshwater_density = 997    #[kg/m^3]
    freshwater_m3_per_hr = freshwater_kg_per_hr / freshwater_density 
    desal_capacity = freshwater_m3_per_hr

    if salinity == "Seawater":
        # SWRO: Sea Water Reverse Osmosis, water >18,000 ppm 
        # Water recovery
        recovery_ratio = 0.5    #https://www.usbr.gov/research/dwpr/reportpdfs/report072.pdf
        feedwater_m3_per_hr = freshwater_m3_per_hr / recovery_ratio

        # Power required
        energy_conversion_factor = 4.0  #[kWh/m^3] SWRO energy_conversion_factor range 2.5 to 4.0 kWh/m^3
                                        #https://www.sciencedirect.com/science/article/pii/S0011916417321057
        desal_power = freshwater_m3_per_hr * energy_conversion_factor


    elif salinity == "Brackish":
        # BWRO: Brakish water Reverse Osmosis, water < 18,000 ppm   
        # Water recovery
        recovery_ratio = 0.75    #https://www.usbr.gov/research/dwpr/reportpdfs/report072.pdf
        feedwater_m3_per_hr = freshwater_m3_per_hr / recovery_ratio

        # Power required
        energy_conversion_factor = 1.5  #[kWh/m^3] BWRO energy_conversion_factor range 1.0 to 1.5 kWh/m^3
                                        #https://www.sciencedirect.com/science/article/pii/S0011916417321057    
                                        
        desal_power = freshwater_m3_per_hr * energy_conversion_factor
    
    else:
        raise Exception("Salinity parameter must be set to Brackish or Seawater")

    # Costing
    # https://www.nrel.gov/docs/fy16osti/66073.pdf
    desal_capex = 32894 * (freshwater_density * desal_capacity / 3600) # [USD]

    desal_opex = 4841 * (freshwater_density * desal_capacity / 3600) # [USD/yr]

    '''Mass and Footprint
    Based on Commercial Industrial RO Systems
    https://www.appliedmembranes.com/s-series-seawater-reverse-osmosis-systems-2000-to-100000-gpd.html
    
    All Mass and Footprint Estimates are estimated from Largest RO System:
    S-308F
    -436 m^3/day
    -6330 kg
    -762 cm (L) x 112 cm (D) x 183 cm (H)

    436 m^3/day = 18.17 m^3/hr = 8.5 m^2, 6330 kg
    1 m^3/hr = .467 m^2, 346.7 kg

    Voltage Codes
    460 or 480v/ 3ph/ 60 Hz
    '''
    desal_mass_kg = freshwater_m3_per_hr * 346.7    #[kg]
    desal_size_m2 = freshwater_m3_per_hr * .467     #[m^2]

    
    return desal_capacity, feedwater_m3_per_hr, desal_power, desal_capex, desal_opex, desal_mass_kg, desal_size_m2

if __name__ == '__main__':
    desal_freshwater_kg_hr = 75000
    salinity = 'Brackish'
    test = RO_desal_eco(desal_freshwater_kg_hr,salinity)
    print(test)