from hybrid.financial.custom_cost_constants import *
from hybrid.layout.pv_design_utils import get_num_modules

"""
Accessor map where values are either the name of the variable within the HybridSimulation class 
or a function with signature fn(HybridSimulation) -> float
"""
BOS_DetailedPVPlant_input_map = {
    (SOLAR, MAIN_EQUIPMENT, PV_MODULES): get_num_modules,
    (SOLAR, MAIN_EQUIPMENT, ITS): "ninverters"
}

class CustomCostModel:
    def __init__(self, bos_config) -> None:
        pass

    def calculate_total_costs(self, bos_data):
        """
        bos_data has data from pv, wind, battery, etc
        """
        pass