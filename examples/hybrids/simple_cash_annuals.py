import numpy as np

def simple_cash_annuals(plant_useful_life, equipment_useful_life, capex, opex, amortization_interest):
   
    # Number of cycles for equipment
    full_cycle = plant_useful_life // equipment_useful_life
    partial_cycle = plant_useful_life % equipment_useful_life
    
    #Payment period for equipment (assume payment period is life of equipment)
    a = amortization_interest
    amortization = []
    
    for i in range(full_cycle*equipment_useful_life):   
        amortization.append(capex * ((a*(1+a)**equipment_useful_life)\
            /(((1+a)**equipment_useful_life)-1)))
    
    #Payment period if plant's useful life is sorter than equipment 
    for i in range(full_cycle*equipment_useful_life, plant_useful_life):
        amortization.append(capex * ((a*(1+a)**(partial_cycle))\
            /(((1+a)**(partial_cycle))-1)))

    # Initialize annual cash flow array for useful life of plant
    opex_annuals = [opex] * plant_useful_life

    cash_flow_annuals = np.add(opex_annuals, amortization)
    # print(cash_flow_annuals)
    return cash_flow_annuals


if __name__ == '__main__': 
    simple_cash_annuals(30,8, 1000,20,0.03)