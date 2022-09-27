import numpy as np
import numpy_financial as npf
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals


def basic_H2_cost_model(electrolyzer_size_mw, useful_life, atb_year, 
    electrical_generation_timeseries, hydrogen_annual_output, tax_credit_USD_kg):
    """
    Basic cost modeling for a PEM electrolyzer.
    Looking at cost projections for PEM electrolyzers over years 2022, 2025, 2030, 2035.
    Electricity costs are calculated outside of hydrogen cost model

    Needs:
    Scaling factor for off-shore electrolysis
    Verifying numbers are appropriate for simplified cash flows
    Verify how H2 PTC works/factors into cash flows
    """


    # Basic information in our analysis
    discount_rate = 0.07
    kw_continuous = electrolyzer_size_mw *1000

    # Capacity factor
    avg_generation = np.mean(electrical_generation_timeseries)  # Avg Generation
        # print("avg_generation: ", avg_generation)
    cap_factor = avg_generation / kw_continuous
    print("cap_factor",cap_factor)

    #Apply PEM Cost Estimates based on year based on GPRA pathway (H2New)
    if atb_year == 2022:
        electrolyzer_capex_kw = 1100     #[$/kW capacity] stack capital cost
        time_between_replacement = 40000    #[hrs] 
    elif atb_year == 2025:
        electrolyzer_capex_kw = 300
        time_between_replacement = 80000    #[hrs]
    elif atb_year == 2030:
        electrolyzer_capex_kw = 150
        time_between_replacement = 80000    #[hrs]
    elif atb_year == 2035:
        electrolyzer_capex_kw = 100
        time_between_replacement = 80000    #[hrs]

    # Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
    # https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf

    # Capital costs provide by Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
    stack_capital_cost = 342   #[$/kW]
    mechanical_bop_cost = 36  #[$/kW] for a compressor
    electrical_bop_cost = 82  #[$/kW] for a rectifier

    # Installed capital cost
    stack_installation_factor = 12/100  #[%] for stack cost 
    elec_installation_factor = 12/100   #[%] and electrical BOP 
    #mechanical BOP install cost = 0%

    # Indirect capital cost as a percentage of installed capital cost
    site_prep = 2/100   #[%]
    engineering_design = 10/100 #[%]
    project_contingency = 15/100 #[%]
    permitting = 15/100     #[%]
    land = 250000   #[$]

    stack_replacment_cost = 15/100  #[% of installed capital cost]
    plant_lifetime = 40    #[years]
    fixed_OM = 0.24     #[$/kg H2]

    program_record = False

    # Chose to use numbers provided by GPRA pathways
    if program_record:
        total_direct_electrolyzer_cost_kw = (stack_capital_cost*(1+stack_installation_factor)) \
            + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))
    else:
        total_direct_electrolyzer_cost_kw = (electrolyzer_capex_kw * (1+stack_installation_factor)) \
            + mechanical_bop_cost + (electrical_bop_cost*(1+elec_installation_factor))

    # Assign CapEx for electrolyzer from capacity based installed CapEx
    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw* electrolyzer_size_mw *1000

    # Add indirect capital costs
    electrolyzer_total_capital_cost = ((site_prep+engineering_design+project_contingency+permitting)\
        *electrolyzer_total_installed_capex) + land

    # O&M costs
    # https://www.sciencedirect.com/science/article/pii/S2542435121003068
    fixed_OM = 12.8 #[$/kW-y]
    property_tax_insurance = 1.5/100    #[% of Cap/y]
    variable_OM = 1.30  #[$/MWh]

    # Amortized refurbishment expense [$/MWh]
    amortized_refurbish_cost = (total_direct_electrolyzer_cost_kw*stack_replacment_cost)\
            *max(((useful_life*8760*cap_factor)/time_between_replacement-1),0)/useful_life/8760/cap_factor*1000

    # Total O&M costs [% of installed cap/year]
    total_OM_costs = ((fixed_OM+(property_tax_insurance*total_direct_electrolyzer_cost_kw))/total_direct_electrolyzer_cost_kw\
        +((variable_OM+amortized_refurbish_cost)/1000*8760*(cap_factor/total_direct_electrolyzer_cost_kw)))

    capacity_based_OM = True
    if capacity_based_OM:
        electrolyzer_OM_cost = electrolyzer_total_installed_capex * total_OM_costs     #Capacity based
    else:   
        electrolyzer_OM_cost = fixed_OM  * hydrogen_annual_output #Production based - likely not very accurate

    # Add in electrolyzer repair schedule (every 7 years)
    # Use if not using time between replacement given in hours 
    # Currently not added into further calculations
    electrolyzer_repair_schedule = []
    counter = 1
    for year in range(0,useful_life):
        if year == 0:
            electrolyzer_repair_schedule = np.append(electrolyzer_repair_schedule, [0])

        elif counter % time_between_replacement == 0:
            electrolyzer_repair_schedule = np.append(electrolyzer_repair_schedule, [1])

        else:
            electrolyzer_repair_schedule = np.append(electrolyzer_repair_schedule, [0])
        counter += 1
    electrolyzer_replacement_costs = electrolyzer_repair_schedule * (stack_replacment_cost* electrolyzer_total_installed_capex)
    # print("H2 replacement costs: ", electrolyzer_replacement_costs)

    # Include Hydrogen PTC from the Inflation Reduction Act (range $0.60 - $3/kg-H2)
    h2_tax_credit = [0] * useful_life
    h2_tax_credit[0:10] = [hydrogen_annual_output* tax_credit_USD_kg] * 10
    print('H2 tax credit',h2_tax_credit)

    # Simple cash annuals
    cf_h2_annuals = - simple_cash_annuals(useful_life, useful_life, electrolyzer_total_capital_cost,\
        electrolyzer_OM_cost, 0.03)

    print("CF H2 Annuals",cf_h2_annuals)

    # Add positive cashflow from tax credit
    cf_h2_annuals = np.add(cf_h2_annuals,h2_tax_credit)

    print('Added H2 ptc with cash flows', cf_h2_annuals)

    return cf_h2_annuals, electrolyzer_total_capital_cost, electrolyzer_OM_cost, electrolyzer_capex_kw, time_between_replacement, h2_tax_credit
