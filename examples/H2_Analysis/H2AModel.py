import math
import pandas as pd
import numpy as np


def H2AModel(cap_factor, avg_daily_H2_production, hydrogen_annual_output, h2a_for_hopp=True, force_system_size=True,
             forced_system_size=50, force_electrolyzer_cost=False, forced_electrolyzer_cost_kw=200, useful_life = 30):

    # ---------------------------------------------------H2A PROCESS FLOW----------------------------------------------------------#


    current_density = 2  # A/cm^2
    voltage = 1.9  # V/cell
    operating_temp = 80  # C
    H2_outlet_pressure = 450  # psi
    cell_active_area = 450  # cm^2
    cellperstack = 150  # cells
    degradation_rate = 1.5  # mV/1000 hrs
    stack_life = 7  # years
    hours_per_stack_life = stack_life * 365 * 24 * cap_factor  # hrs/life
    degradation_rate_Vperlife = hours_per_stack_life * degradation_rate / 1000  # V/life
    stack_degradation_oversize = 0.13  # factor
    peak_daily_production_rate = avg_daily_H2_production * (1 + stack_degradation_oversize)  # kgH2/day

    total_active_area = math.ceil(
        (avg_daily_H2_production / 2.02 * 1000 / 24 / 3600) * 2 * 96485 / current_density / (100 ** 2))  # m^2
    total_active_area_degraded = math.ceil(
        (peak_daily_production_rate / 2.02 * 1000 / 24 / 3600) * 2 * 96485 / current_density / (100 ** 2))  # m^2

    stack_electrical_usage = 50.4  # kWh/kgH2
    BoP_electrical_usage = 5.1  # kWh/kgH2
    total_system_electrical_usage = 55.5  # kWh/kg H2

    if forced_system_size:
        total_system_input = forced_system_size  # MW
        stack_input_power = (stack_electrical_usage/ total_system_electrical_usage) * forced_system_size  # MW
    else:
        total_system_input = total_system_electrical_usage / 24 * peak_daily_production_rate / 1000  # MW
        stack_input_power = stack_electrical_usage / 24 * peak_daily_production_rate / 1000  # MW

    process_water_flowrate = 3.78

    system_unit_cost = 1.3 * 300/342 # $/cm^2
    stack_system_cost = system_unit_cost / (current_density * voltage) * 1000  # $/kW
    mechanical_BoP_unit_cost = 76  # kWhH2/day
    mechanical_BoP_cost = 76 * peak_daily_production_rate / stack_input_power / 1000  # $/kW
    electrical_BoP_cost = 82  # $/kW
    total_system_cost_perkW = stack_system_cost + mechanical_BoP_cost + electrical_BoP_cost  # $/kW
    total_system_cost_perkW = total_system_cost_perkW
    if force_electrolyzer_cost:
        total_system_cost = forced_electrolyzer_cost_kw * stack_input_power * 1000
    else:
        total_system_cost = total_system_cost_perkW * stack_input_power * 1000  # $


    # -------------------------------------------------CAPITAL COST--------------------------------------------------------------#


    gdpdef = {'Year': [2015, 2016, 2017, 2018, 2019, 2020], 'CEPCI': [104.031, 104.865, 107.010, 109.237, 111.424,
                                                                      113.415]}  # GDPDEF (2012=100), https://fred.stlouisfed.org/series/GDPDEF/
    CEPCI = pd.DataFrame(data=gdpdef)  # Deflator Table

    pci = {'Year': [2015, 2016, 2017, 2018, 2019, 2020],
           'PCI': [556.8, 541.7, 567.5, 603.1, 607.5, 610]}  # plant cost index, Chemical Engineering Magazine
    CPI = pd.DataFrame(data=pci)

    baseline_plant_design_capacity = avg_daily_H2_production
    basis_year_for_capital_cost = 2016
    current_year_for_capital_cost = 2016
    CEPCI_inflator = int((CEPCI.loc[CEPCI['Year'] == current_year_for_capital_cost, 'CEPCI']).iloc[0]) / int(
        (CEPCI.loc[CEPCI['Year'] == basis_year_for_capital_cost, 'CEPCI']).iloc[0])
    consumer_price_inflator = int((CPI.loc[CPI['Year'] == current_year_for_capital_cost, 'PCI']).iloc[0]) / int(
        (CPI.loc[CPI['Year'] == basis_year_for_capital_cost, 'PCI']).iloc[0])  # lookup

    # --------------------------CAPITAL INVESTMENT---------------------------------#
    # ----Inputs required in basis year (2016$)----#

    baseline_uninstalled_stack_capital_cost = CEPCI_inflator * consumer_price_inflator * system_unit_cost * total_active_area_degraded * 100 ** 2  # ($2016)
    stack_installation_factor = 1.12
    baseline_installed_stack_capital_cost = stack_installation_factor * baseline_uninstalled_stack_capital_cost

    baseline_uninstalled_mechanical_BoP_cost = CEPCI_inflator * consumer_price_inflator * mechanical_BoP_unit_cost * peak_daily_production_rate  # ($2016)
    mechanical_BoP_installation_factor = 1
    baseline_installed_mechanical_BoP_cost = mechanical_BoP_installation_factor * baseline_uninstalled_mechanical_BoP_cost

    baseline_uninstalled_electrical_BoP_cost = CEPCI_inflator * consumer_price_inflator * electrical_BoP_cost * stack_input_power * 1000  # ($2016)
    electrical_BoP_installation_factor = 1.12
    baseline_installed_electrical_BoP_cost = electrical_BoP_installation_factor * baseline_uninstalled_electrical_BoP_cost

    baseline_total_installed_cost = baseline_installed_stack_capital_cost + baseline_installed_mechanical_BoP_cost + baseline_installed_electrical_BoP_cost

    # ------------------------------------------------PLANT SCALING-------------------------------------------------------------------#

    scale_ratio = 1  # ratio of new design capacity to baseline design capacity (linear scaling)
    scale_factor = 1  # rato of total scaled installed capital cost to total baseline installed capital cost (exponential scaling)
    default_scaling_factor_exponent = 1  # discrepancy
    lower_limit_for_scaling_capacity = 20000  # kgH2/day
    upper_limit_for_scaling_capacity = 200000  # kgH2/day

    scaled_uninstalled_stack_capital_cost = baseline_uninstalled_stack_capital_cost ** default_scaling_factor_exponent
    scaled_installed_stack_capital_cost = scaled_uninstalled_stack_capital_cost * stack_installation_factor

    scaled_uninstalled_mechanical_BoP_cost = baseline_uninstalled_mechanical_BoP_cost ** default_scaling_factor_exponent
    scaled_installed_mechanical_BoP_cost = scaled_uninstalled_mechanical_BoP_cost * mechanical_BoP_installation_factor

    scaled_uninstalled_electrical_BoP_cost = baseline_uninstalled_electrical_BoP_cost ** default_scaling_factor_exponent
    scaled_installed_electrical_BoP_cost = scaled_uninstalled_electrical_BoP_cost * electrical_BoP_installation_factor

    scaled_total_installed_cost = scaled_installed_stack_capital_cost + scaled_installed_mechanical_BoP_cost + scaled_installed_electrical_BoP_cost

    # -------------------------------------------------------H2A INPUT-------------------------------------------------------------#

    # --------------------------TECHNICAL OPERATING PARAMETERS AND SPECIFICATIONS---------------------------#

    operating_capacity_factor = cap_factor
    plant_design_capacity = peak_daily_production_rate  # kgH2/day
    plant_daily_output = plant_design_capacity  # kg/day
    plant_annual_output = hydrogen_annual_output  # kg/year

    # -----------------------------------FINANCIAL INPUT VALUES---------------------------------------------#

    reference_year = 2016
    assumed_startup_year = 2015
    basis_year = 2016
    length_of_construction_period = 1  # year
    startup_time = length_of_construction_period
    percent_capital_spent_year_1 = 100 / 100  # percent
    percent_Capital_Spent_year_2 = 0 / 100  # percent
    percent_Capital_Spent_year_3 = 0 / 100  # percent
    percent_Capital_Spent_year_4 = 0 / 100  # percent
    plant_life = useful_life  # years
    analysis_period = useful_life  # years
    depreciation_schedule_length = 20  # years
    depreciation_type = 'MACRS'
    percent_equity_financing = 40 / 100  # percent
    debt_financing = 60 / 100  # percent
    interest_rate_on_debt = 3.7 / 100  # percent
    debt_period = 'Constant Debt'
    percent_fixed_operating_cost_during_startup = 75 / 100  # percent
    percent_revenue_during_startup = 50 / 100  # percent
    percent_variable_operating_cost_during_startup = 75 / 100  # percent
    decom_pct = 10 / 100  # percent of depreciable capital investment
    salvage_pct = 10 / 100  # percent of total capital investment
    inflation_rate = 1.9 / 100  # percent
    inflation_factor = (1 + inflation_rate) ** (assumed_startup_year - reference_year)
    after_tax_real_IRR = 8 / 100  # percent
    nominal_IRR = ((1 + after_tax_real_IRR) * (1 + inflation_rate)) - 1
    state_taxes = 6 / 100  # percent
    federal_taxes = 21 / 100  # percent
    total_tax_rate = federal_taxes + state_taxes * (1 - federal_taxes)  # percent
    working_capital = 15 / 100  # percent of yearly change in operating cost
    unplanned_replacement_cost_factor = 0.005
    tax_credit = 0

    # ------------------------Energy Feedstock, Ulilities, and Byproducts---------------------------------------#

    price_table_reference = 'AEO_2017_Reference_Case'

    # Add Feedstock Info
    feedstock_type = '--------'
    feedstock_price_conversion_factor = 0  # GJ/kWh
    feedstock_price_in_startup_year = 0  # ($2016)/kWh (LookUp)
    feedstock_usage = 55.5  # kWh/kgH2 (LookUp)
    total_energy_feedstock_unitcost = 0.00
    feedstock_cost_in_startup_year = plant_annual_output * feedstock_usage * feedstock_price_in_startup_year

    total_energy_feedstock_cost = feedstock_cost_in_startup_year  # Add all feedstock costs in startup year

    # Add Utilities Info
    utilities_type = 'Industrial Electricity'
    electricity_LHV = 0.0036  # GJ/kWh
    utilities_price_in_startup_year = 0.00  # ($2016)/kWh (LookUp)
    electricity_usage = 55.5  # kWh/kgH2 (LookUp)
    utilities_cost_in_startup_year = plant_annual_output * electricity_usage * utilities_price_in_startup_year

    total_energy_utilities_cost = utilities_cost_in_startup_year  # Add all utilities costs in startup year

    # Add ByProducts Info
    byproduct_type = '--------------'
    byproduct_price_conversion_factor = 0  # GJ/kWh
    byproduct_price_in_startup_year = 0  # ($2016)/kWh (LookUp)
    byproduct_production = 0  # kWh/kgH2 (LookUp)
    total_byproduct_unitprice = 0
    byproduct_income_in_startup_year = plant_annual_output * byproduct_production * byproduct_price_in_startup_year

    total_energy_byproduct_credit = byproduct_income_in_startup_year  # Add all byproduct credits in startup year

    # --------------------------------------Capital Costs--------------------------------------------------------#

    H2A_total_direct_capital_cost = int(scaled_total_installed_cost)
    cost_scaling_factor = 1  # combined plant scaling and escalation factor

    # ------------Indirect Depreciable Capital Costs---------------------#
    site_preparation = 0.02 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor
    engineering_design = 0.1 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor
    project_contingency = 0.15 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor
    upfront_permitting_cost = 0.15 * H2A_total_direct_capital_cost / (
            CEPCI_inflator * consumer_price_inflator) * cost_scaling_factor

    total_depreciable_costs = int(
        H2A_total_direct_capital_cost + site_preparation + engineering_design + project_contingency + upfront_permitting_cost)

    # ------------Non Depreciable Capital Costs---------------------#
    cost_of_land = 50000  # ($2016)/acre
    land_required = 5  # acres
    land_cost = cost_of_land * land_required
    other_nondepreciable_cost = 0

    total_nondepreciable_costs = land_cost + other_nondepreciable_cost

    total_capital_costs = total_depreciable_costs + total_nondepreciable_costs

    # --------------------------------------Fixed Operating Costs------------------------------------------------#
    total_plant_staff = 10  # number of FTEs employed by plant
    burdened_labor_cost = 50  # including overhead ($/man-hr)
    labor_cost = total_plant_staff * burdened_labor_cost * 2080  # ($2016)/year

    GA_rate = 20  # percent labor cos (general and admin)
    GA_cost = labor_cost * (GA_rate / 100)  # $/year
    licensing_permits_fees = 0  # $/year
    propertytax_insurancerate = 2  # percent of total capital investment per year
    propertytax_insurancecost = total_capital_costs * (propertytax_insurancerate / 100)  # $/year
    rent = 0  # $/year
    material_costs_for_maintenance = 0.03 * H2A_total_direct_capital_cost / (CEPCI_inflator * consumer_price_inflator)
    production_maintenance_and_repairs = 0  # $/year
    other_fees = 0  # $/year
    other_fixed_OM_costs = 0  # $/year

    total_fixed_operating_costs = int(labor_cost + GA_cost + licensing_permits_fees + propertytax_insurancecost + rent \
                                      + material_costs_for_maintenance + production_maintenance_and_repairs + other_fees + other_fixed_OM_costs)

    # --------------------------------------Variable Operating Costs----------------------------------------------#

    # ------------------Other Material and Byproduct---------------------#

    Material_1 = 'Processed Water'  # feed
    processed_water_cost = 0.002375  # ($2016)/gal
    water_usage_per_kgH2 = 3.78  # usageperkgH2
    feedcost_in_startup_year = plant_annual_output * processed_water_cost * water_usage_per_kgH2  # lookup

    total_nonE_utility_material_costs = feedcost_in_startup_year  # Add all feed or utlity start-up year costs
    total_nonE_byproduct_credit = 0  # Add all byproduct start-up year credits

    total_feedstock_cost = total_energy_feedstock_cost
    total_utility_cost = total_energy_utilities_cost + total_nonE_utility_material_costs
    total_byproduct_credit = total_energy_byproduct_credit + total_nonE_byproduct_credit

    # -----------------Other Variable Operating Costs---------------------#

    other_variable_operating_costs = 0  # ($2016)/year (e.g. environmental surchages)
    other_material_costs = 0  # ($2016)/year
    waste_treatment_costs = 0  # ($2016)/year
    solid_waste_disposal_costs = 0  # ($2016)/year
    total_unplanned_replacement_capital_cost_factor = 0.5  # percent of total direct depreciable cost (costs/year)
    total_unplanned_replacement_capital_cost = total_depreciable_costs * (
            total_unplanned_replacement_capital_cost_factor / 100)  # ($2016)/year

    total_variable_operating_costs = (
            total_feedstock_cost + total_utility_cost + total_byproduct_credit - total_byproduct_credit +
            other_variable_operating_costs + other_material_costs + waste_treatment_costs +
            solid_waste_disposal_costs + total_unplanned_replacement_capital_cost_factor + total_unplanned_replacement_capital_cost)  # ($2016)/year

    # ---------------------------------------------------RESULTS----------------------------------------------------------#

    # --------------------------------------Hydorgen Energy Constants----------------------------------------------#

    H2_LHV_MJkg = 120.21  # MJ/kg
    H2_LHV_Btulb = 51.682  # Btu/lb
    mmBTU_to_GJ = 1.055

    # --------------------------------------DCF Calculation Inputs----------------------------------------------#

    # PROCESS
    actual_hydrogen_produced = plant_annual_output  # kg/yr
    actual_hydrogen_energy_produced_Btu = actual_hydrogen_produced * H2_LHV_MJkg / 1000 / mmBTU_to_GJ  # MMBtu(LHV)/yr
    actual_hydrogen_energy_produced_MJ = actual_hydrogen_produced * H2_LHV_MJkg

    # Additional FINANCIALS

    inflated_fixed = total_fixed_operating_costs * inflation_factor
    total_depreciable_costs_inflated = (total_capital_costs - total_nondepreciable_costs) * inflation_factor
    decom = decom_pct * total_depreciable_costs * inflation_factor
    salvage_value = salvage_pct * inflation_factor * (total_depreciable_costs + total_nondepreciable_costs)
    inflated_other_rawmaterial = 0
    inflated_othervar = 0
    H2_price_nominal = 4.74027691  # not included

    # --------------------------------------DCF Calculation OUTPUT----------------------------------------------

    Initial_Data = {
        'Year': np.arange(assumed_startup_year - length_of_construction_period, assumed_startup_year + plant_life),
        'Analysis Year': np.arange(plant_life + 1)}
    df = pd.DataFrame(data=Initial_Data)

    df['Inflation Price Increase Factor'] = (1 + inflation_rate) ** (df['Year'] - assumed_startup_year)


    df['Revenue from Hydrogen Sales'] = 0
    df['Revenue from Hydrogen Sales'][df['Analysis Year'] == 0] = 0
    df['Revenue from Hydrogen Sales'][df['Analysis Year'] == 1] = H2_price_nominal * df[
        'Inflation Price Increase Factor'] * plant_annual_output * percent_revenue_during_startup
    df['Revenue from Hydrogen Sales'][(df['Analysis Year'] == 1) & (startup_time < 1)] = df['Inflation Price Increase Factor'] * H2_price_nominal * plant_annual_output * startup_time * percent_revenue_during_startup \
                                + H2_price_nominal * plant_annual_output * (1 - startup_time)
    df['Revenue from Hydrogen Sales'][df['Analysis Year'] > 1] = H2_price_nominal * plant_annual_output * df[
        'Inflation Price Increase Factor']


    df['Debt Financed Initial Depreciable Capital'] = 0
    df['Debt Financed Initial Depreciable Capital'][df['Analysis Year'] == 0] = -df[
        'Inflation Price Increase Factor'] * debt_financing * total_depreciable_costs * inflation_factor
    df['Debt Financed Initial Depreciable Capital'][df['Analysis Year'] >= 1] = 0


    df['Initial Equity Depreciable Capital'] = 0
    df['Initial Equity Depreciable Capital'][df['Analysis Year'] < 1] = -df[
        'Inflation Price Increase Factor'] * total_depreciable_costs_inflated * percent_equity_financing * percent_capital_spent_year_1




    df['Yearly Replacement Cost'] = 0
    df['Yearly Replacement Cost'][
        df['Analysis Year'] >= 1] = -unplanned_replacement_cost_factor * total_depreciable_costs * df[
        'Inflation Price Increase Factor'] * ((1 + inflation_rate) ** (assumed_startup_year - reference_year))
    df['Yearly Replacement Cost'][
        (df['Analysis Year'] == 8) | (df['Analysis Year'] == 15) | (df['Analysis Year'] == 22) | (
                df['Analysis Year'] == 29) | (df['Analysis Year'] == 36)] = -df[
        'Inflation Price Increase Factor'] * ((unplanned_replacement_cost_factor * total_depreciable_costs +
                                               0.15 * H2A_total_direct_capital_cost) * (1 + inflation_rate) ** (
                                                      assumed_startup_year - reference_year))  # /(CPInfactor*CEPCIinflator)




    df['Other Non-Depreciable Capital Cost'] = 0
    df['Other Non-Depreciable Capital Cost'][df['Analysis Year'] == 0] = -total_nondepreciable_costs * df[
        'Inflation Price Increase Factor'] * inflation_factor




    df['Salvage Value'] = 0
    df['Salvage Value'][df['Analysis Year'] == analysis_period] = df['Inflation Price Increase Factor'] * salvage_value




    df['Decommissioning Costs'] = 0
    df['Decommissioning Costs'][df['Analysis Year'] == analysis_period] = -df['Inflation Price Increase Factor'] * decom




    df['Fixed Operating Cost'] = 0
    df['Fixed Operating Cost'][df['Analysis Year'] >= 1] = -inflated_fixed * df['Inflation Price Increase Factor']
    df['Fixed Operating Cost'][(df['Analysis Year'] == 1) & (startup_time < 1)] = -(
            inflated_fixed * percent_fixed_operating_cost_during_startup * startup_time + (
            inflated_fixed * (1 - startup_time)))
    df['Fixed Operating Cost'][df['Analysis Year'] == startup_time] = -(
            inflated_fixed * percent_fixed_operating_cost_during_startup * df['Inflation Price Increase Factor'])




    df['Feedstock Cost'] = 0
    df['Feedstock Cost'][(df['Analysis Year'] == 1) & (startup_time < 1)] = -(
            total_energy_feedstock_unitcost * plant_annual_output * df[
        'Inflation Price Increase Factor'] * startup_time * percent_variable_operating_cost_during_startup + total_energy_feedstock_unitcost * plant_annual_output *
            df['Inflation Price Increase Factor'] * (1 - startup_time))
    df['Feedstock Cost'][df['Analysis Year'] <= startup_time] = -total_energy_feedstock_unitcost * plant_annual_output * \
                                                                df[
                                                                    'Inflation Price Increase Factor'] * percent_variable_operating_cost_during_startup
    df['Feedstock Cost'][df['Analysis Year'] >= 1] = -total_energy_feedstock_unitcost * plant_annual_output * df[
        'Inflation Price Increase Factor']




    df['Other Raw Material Cost'] = 0
    df['Other Raw Material Cost'][(df['Analysis Year'] == 1) & (startup_time < 1)] = -(inflated_other_rawmaterial * df[
        'Inflation Price Increase Factor'] * startup_time * percent_variable_operating_cost_during_startup + inflated_other_rawmaterial *
                                                                                       df[
                                                                                           'Inflation Price Increase Factor'] * (
                                                                                               1 - startup_time))
    df['Other Raw Material Cost'][df['Analysis Year'] <= startup_time] = -inflated_other_rawmaterial * df[
        'Inflation Price Increase Factor'] * percent_variable_operating_cost_during_startup
    df['Other Raw Material Cost'][df['Analysis Year'] >= 1] = -inflated_other_rawmaterial * df[
        'Inflation Price Increase Factor']




    df['Revenue from Byproduct Sales'] = 0
    df['Revenue from Byproduct Sales'][(df['Analysis Year'] == 1) & (startup_time < 1)] = (
            total_byproduct_unitprice * plant_annual_output * df[
        'Inflation Price Increase Factor'] * startup_time * percent_variable_operating_cost_during_startup + total_energy_feedstock_unitcost * plant_annual_output *
            df['Inflation Price Increase Factor'] * (1 - startup_time))
    df['Revenue from Byproduct Sales'][
        df['Analysis Year'] <= startup_time] = total_byproduct_unitprice * plant_annual_output * df[
        'Inflation Price Increase Factor'] * percent_variable_operating_cost_during_startup
    df['Revenue from Byproduct Sales'][df['Analysis Year'] >= 1] = total_byproduct_unitprice * plant_annual_output * df[
        'Inflation Price Increase Factor']




    Process_Water_Data = {'Year': np.arange(2014, 2061), 'Price ($(2016)/gal)': [0.002374951] * 47}
    Process_Water_Price = pd.DataFrame(data=Process_Water_Data)
    Process_Water_Price['Unit Cost'] = inflation_factor * water_usage_per_kgH2 * Process_Water_Price[
        'Price ($(2016)/gal)']
    Process_Water_Price = Process_Water_Price[(Process_Water_Price.Year >= assumed_startup_year - 1) & (
            Process_Water_Price.Year < assumed_startup_year + analysis_period)]
    Process_Water_Price.reset_index(drop=True, inplace=True)

    Industrial_Electricity_Data = {'Year': np.arange(2014, 2061),
                                   'Price ($(2012)/GJ LHV)': [20.19068, 19.43906, 18.77777252, 19.5374937444079,
                                                              19.0946888127265, 19.3408804840691, 19.7007267860663,
                                                              19.8092319270257, 20.4486283192551, 20.5134381492546,
                                                              20.4593613988262, \
                                                              20.9506225261977, 21.1633846923765, 21.2853222956886,
                                                              21.4236239592969, 21.5322826466083, 21.6077715306107,
                                                              21.6351454330386, 21.5857850199421, 21.508628925858,
                                                              21.4461184998711, \
                                                              21.5219523892571, 21.6002041597792, 21.5838723252605,
                                                              21.5886663835853, 21.6053745014483, 21.5547155790783,
                                                              21.4913246311096, 21.5162512700748, 21.5819141353634,
                                                              21.6632320938415, 21.7423672297963, \
                                                              21.8228245704493, 21.8950245295036, 21.97623917593,
                                                              22.0466032134787, 22.1148194977328, 22.1348766528058,
                                                              22.1549519988241, 22.1750455522859, 22.1951573297046,
                                                              22.2152873476085, 22.2374072845512, \
                                                              22.2595492464896, 22.2817132553544, 22.3038993330976,
                                                              22.3261075016936]}
    Industrial_Electricity_Price = pd.DataFrame(data=Industrial_Electricity_Data)
    if h2a_for_hopp:
        Industrial_Electricity_Price['Unit Cost'] = 0
    else:
        Industrial_Electricity_Price['Unit Cost'] = inflation_factor * electricity_usage * electricity_LHV * \
                                                    Industrial_Electricity_Price['Price ($(2012)/GJ LHV)']
    Industrial_Electricity_Price = Industrial_Electricity_Price[
        (Industrial_Electricity_Price.Year >= assumed_startup_year - 1) & (
                Industrial_Electricity_Price.Year < assumed_startup_year + analysis_period)]
    Industrial_Electricity_Price.reset_index(drop=True, inplace=True)

    df['Other Variable Operating Costs'] = 0
    df['Other Variable Operating Costs'][(df['Analysis Year'] == 1) & (startup_time < 1)] = -(((
                                                                                                       Industrial_Electricity_Price[
                                                                                                           'Unit Cost'] +
                                                                                                       Process_Water_Price[
                                                                                                           'Unit Cost']) * plant_annual_output + inflated_othervar) *
                                                                                              df[
                                                                                                  'Inflation Price Increase Factor'] * startup_time * percent_variable_operating_cost_during_startup + \
                                                                                              ((
                                                                                                       Industrial_Electricity_Price[
                                                                                                           'Unit Cost'] +
                                                                                                       Process_Water_Price[
                                                                                                           'Unit Cost']) * plant_annual_output + inflated_othervar) *
                                                                                              df[
                                                                                                  'Inflation Price Increase Factor'] * (
                                                                                                      1 - startup_time))
    df['Other Variable Operating Costs'][df['Analysis Year'] <= startup_time] = -((Industrial_Electricity_Price[
                                                                                       'Unit Cost'] +
                                                                                   Process_Water_Price[
                                                                                       'Unit Cost']) * plant_annual_output + inflated_othervar) * \
                                                                                df[
                                                                                    'Inflation Price Increase Factor'] * percent_variable_operating_cost_during_startup
    df['Other Variable Operating Costs'][df['Analysis Year'] > 1] = -df['Inflation Price Increase Factor'] * ((
                                                                                                                      Industrial_Electricity_Price[
                                                                                                                          'Unit Cost'] +
                                                                                                                      Process_Water_Price[
                                                                                                                          'Unit Cost']) * plant_annual_output + inflated_othervar)
    df['Other Variable Operating Costs'][df['Analysis Year'] == 0] = 0





    df['Working Capital Reserve'] = 0
    i = 1
    for i in range(i, len(df) - 1):
        df.loc[i, 'Working Capital Reserve'] = working_capital * ((df.loc[i, 'Fixed Operating Cost'] + df.loc[
            i, 'Feedstock Cost'] + df.loc[i, 'Other Raw Material Cost'] + df.loc[i, 'Revenue from Byproduct Sales'] +
                                                                   df.loc[i, 'Other Variable Operating Costs']) \
                                                                  - (df.loc[i - 1, 'Fixed Operating Cost'] + df.loc[
                    i - 1, 'Feedstock Cost'] + df.loc[i - 1, 'Other Raw Material Cost'] + df.loc[
                                                                         i - 1, 'Revenue from Byproduct Sales'] +
                                                                     df.loc[i - 1, 'Other Variable Operating Costs']))
    df['Working Capital Reserve'][df['Analysis Year'] == analysis_period] = -df.loc[0:analysis_period,
                                                                             'Working Capital Reserve'].sum()




    df['Debt Interest'] = 0
    df['Debt Interest'] = df.loc[0, 'Debt Financed Initial Depreciable Capital'] * interest_rate_on_debt




    df['Pre-Depreciation Income'] = 0
    df['Pre-Depreciation Income'] = df['Revenue from Hydrogen Sales'] + df['Salvage Value'] + df[
        'Decommissioning Costs'] + df['Fixed Operating Cost'] \
                                    + df['Feedstock Cost'] + df['Other Raw Material Cost'] + df[
                                        'Revenue from Byproduct Sales'] + df['Other Variable Operating Costs'] + df[
                                        'Debt Interest']




    # THREE_YEAR = [0.3333, 0.4445, 0.1481, 0.0741]
    # FIVE_YEAR = [0.2, 0.32, 0.1920, 0.1152, 0.1152, 0.0576]
    # SEVEN_YEAR = [0.1429,0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]
    # TEN_YEAR = [0.1, 0.18, 0.144, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655, 0.0328]
    # FIFTHEEN_YEAR = [0.05, 0.095, 0.0855, 0.077, 0.0693, 0.0623, 0.059, 0.059, 0.0591, 0.059, 0.0591, 0.059, 0.0591, 0.059, 0.0591, 0.0295]
    TWENTY_YEAR = [0.0375, 0.07219, 0.06677, 0.06177, 0.05713, 0.05285, 0.04888, 0.04522, 0.04462, 0.04461, 0.04462,
                   0.04461, 0.04462, 0.04461, 0.04462, 0.04461, 0.04462, 0.04461, 0.04462, 0.04461, 0.02231]

    Depr_Initial_Data = {'Analysis Year': np.arange(plant_life + 21)}
    Depreciation_Table = pd.DataFrame(data=Depr_Initial_Data,
                                      columns=['Annual Depreciable Capital', 'Analysis Year', '1', '2', '3', '4', '5',
                                               '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                               '19', '20', '21'])
    Depreciation_Table['Annual Depreciable Capital'] = 0
    Depreciation_Table['Annual Depreciable Capital'][Depreciation_Table['Analysis Year'] == 1] = -(
            df.loc[0, 'Debt Financed Initial Depreciable Capital'] + df[
        'Initial Equity Depreciable Capital'].sum()) - df['Yearly Replacement Cost']
    Depreciation_Table['Annual Depreciable Capital'][
        (Depreciation_Table['Analysis Year'] > 1) & (Depreciation_Table['Analysis Year'] < plant_life + 1)] = -df[
        'Yearly Replacement Cost']
    Depreciation_Table['Annual Depreciable Capital'][Depreciation_Table['Analysis Year'] > plant_life] = 0

    Depreciation_Table['1'] = 0
    Depreciation_Table['1'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           0]
    Depreciation_Table['2'] = 0
    Depreciation_Table['2'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           1]
    Depreciation_Table['3'] = 0
    Depreciation_Table['3'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           2]
    Depreciation_Table['4'] = 0
    Depreciation_Table['4'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           3]
    Depreciation_Table['5'] = 0
    Depreciation_Table['5'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           4]
    Depreciation_Table['6'] = 0
    Depreciation_Table['6'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           5]
    Depreciation_Table['7'] = 0
    Depreciation_Table['7'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           6]
    Depreciation_Table['8'] = 0
    Depreciation_Table['8'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           7]
    Depreciation_Table['9'] = 0
    Depreciation_Table['9'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                           'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                           8]
    Depreciation_Table['10'] = 0
    Depreciation_Table['10'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            9]
    Depreciation_Table['11'] = 0
    Depreciation_Table['11'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            10]
    Depreciation_Table['12'] = 0
    Depreciation_Table['12'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            11]
    Depreciation_Table['13'] = 0
    Depreciation_Table['13'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            12]
    Depreciation_Table['14'] = 0
    Depreciation_Table['14'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            13]
    Depreciation_Table['15'] = 0
    Depreciation_Table['15'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            14]
    Depreciation_Table['16'] = 0
    Depreciation_Table['16'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            15]
    Depreciation_Table['17'] = 0
    Depreciation_Table['17'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            16]
    Depreciation_Table['18'] = 0
    Depreciation_Table['18'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            17]
    Depreciation_Table['19'] = 0
    Depreciation_Table['19'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            18]
    Depreciation_Table['20'] = 0
    Depreciation_Table['20'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            19]
    Depreciation_Table['21'] = 0
    Depreciation_Table['21'][Depreciation_Table['Analysis Year'] > 0] = Depreciation_Table[
                                                                            'Annual Depreciable Capital'] * TWENTY_YEAR[
                                                                            20]

    Depreciation_Table['Depreciation Charge'] = 0
    for i in range(1, useful_life+21):
        Depreciation_Table.loc[i, 'Depreciation Charge'] = Depreciation_Table.loc[i, '1'] + Depreciation_Table.loc[
            i - 1, '2'] + Depreciation_Table.loc[(i - 2) if (i - 2) > 0 else 0, '3'] + Depreciation_Table.loc[
                                                               (i - 3) if (i - 3) > 0 else 0, '4'] + \
                                                           Depreciation_Table.loc[(i - 4) if (i - 4) > 0 else 0, '5'] + \
                                                           Depreciation_Table.loc[(i - 5) if (i - 5) > 0 else 0, '6'] + \
                                                           Depreciation_Table.loc[(i - 6) if (i - 6) > 0 else 0, '7'] + \
                                                           Depreciation_Table.loc[(i - 7) if (i - 7) > 0 else 0, '8'] + \
                                                           Depreciation_Table.loc[(i - 8) if (i - 8) > 0 else 0, '9'] + \
                                                           Depreciation_Table.loc[(i - 9) if (i - 9) > 0 else 0, '10'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 10) if (i - 10) > 0 else 0, '11'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 11) if (i - 11) > 0 else 0, '12'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 12) if (i - 12) > 0 else 0, '13'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 13) if (i - 13) > 0 else 0, '14'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 14) if (i - 14) > 0 else 0, '15'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 15) if (i - 15) > 0 else 0, '16'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 16) if (i - 16) > 0 else 0, '17'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 17) if (i - 17) > 0 else 0, '18'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 18) if (i - 18) > 0 else 0, '19'] + \
                                                           Depreciation_Table.loc[
                                                               (i - 19) if (i - 19) > 0 else 0, '20'] + \
                                                           Depreciation_Table.loc[(i - 20) if (i - 20) > 0 else 0, '21']




    df['Depreciation Charge'] = 0
    df['Depreciation Charge'][df['Analysis Year'] < plant_life] = -Depreciation_Table['Depreciation Charge']
    df['Depreciation Charge'][df['Analysis Year'] == plant_life] = -Depreciation_Table.loc[
        useful_life, 'Depreciation Charge'] - Depreciation_Table.loc[useful_life+1:useful_life+21, 'Depreciation Charge'].sum()




    df['Taxable Income'] = 0
    df['Taxable Income'] = df['Pre-Depreciation Income'] + df['Depreciation Charge']




    df['Tax Rate (%)'] = total_tax_rate





    df['Total Taxes'] = 0
    df['Total Taxes'] = -df['Taxable Income'] * df['Tax Rate (%)']

    df['After Income Tax'] = 0
    df['After Income Tax'] = df['Pre-Depreciation Income'] + df['Total Taxes']


    df['Principal Payment'] = 0
    df['Principal Payment'][df['Analysis Year'] >= 1] = 0
    df['Principal Payment'][df['Analysis Year'] == analysis_period] = df.loc[
        0, 'Debt Financed Initial Depreciable Capital']



    df['After-Tax Post-Depreciation Cash Flow'] = 0
    df['After-Tax Post-Depreciation Cash Flow'] = df['Initial Equity Depreciable Capital'] + df[
        'Yearly Replacement Cost'] + df['Working Capital Reserve'] \
                                                  + df['Other Non-Depreciable Capital Cost'] + df[
                                                      'Pre-Depreciation Income'] + df['Total Taxes'] + df[
                                                      'Principal Payment']



    df['Cumulative Cash Flow'] = 0
    df['Cumulative Cash Flow'][df['Analysis Year'] == 0] = df.loc[0, 'After-Tax Post-Depreciation Cash Flow']
    i = 1
    for i in range(i, len(df)):
        df.loc[i, 'Cumulative Cash Flow'] = df.loc[i, 'After-Tax Post-Depreciation Cash Flow'] + df.loc[
            i - 1, 'Cumulative Cash Flow']

    df['Pre-Tax Cash Flow'] = 0
    df['Pre-Tax Cash Flow'] = df['Initial Equity Depreciable Capital'] + df['Yearly Replacement Cost'] + df[
        'Working Capital Reserve'] \
                              + df['Other Non-Depreciable Capital Cost'] + df['Pre-Depreciation Income'] + df[
                                  'Principal Payment']


    df['H2 Sales (kg/year)'] = 0
    df['H2 Sales (kg/year)'][(df['Analysis Year'] == 1) & (
            startup_time < 1)] = plant_annual_output * startup_time * percent_revenue_during_startup + plant_annual_output * (
            1 - startup_time)
    df['H2 Sales (kg/year)'][df['Analysis Year'] == startup_time] = plant_annual_output * percent_revenue_during_startup
    df['H2 Sales (kg/year)'][df['Analysis Year'] > 1] = plant_annual_output

    df.loc['TOTAL'] = df.sum(numeric_only=True, axis=0)


    def npv(rate, values):
        npv = (values / ((1 + rate) ** np.arange(1, len(values) + 1))).sum(axis=0)
        return npv


    df.loc['NPV of Cashflow', 'Revenue from Hydrogen Sales'] = npv(nominal_IRR, (df['Revenue from Hydrogen Sales'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Revenue from Hydrogen Sales'] = (df['Revenue from Hydrogen Sales'].tolist())[0] + npv(
        nominal_IRR, (df['Revenue from Hydrogen Sales'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Debt Financed Initial Depreciable Capital'] = npv(nominal_IRR, (df['Debt Financed Initial Depreciable Capital'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Debt Financed Initial Depreciable Capital'] = (df['Debt Financed Initial Depreciable Capital'].tolist())[0] + npv(nominal_IRR, (df[ \
        'Debt Financed Initial Depreciable Capital'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Initial Equity Depreciable Capital'] = npv(nominal_IRR, (df['Initial Equity Depreciable Capital'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Initial Equity Depreciable Capital'] = (df['Initial Equity Depreciable Capital'].tolist())[
                                                                            0] + npv(nominal_IRR, (df['Debt Financed Initial Depreciable Capital'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Yearly Replacement Cost'] = npv(nominal_IRR, (df['Yearly Replacement Cost'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Yearly Replacement Cost'] = (df['Yearly Replacement Cost'].tolist())[0] + npv(nominal_IRR,
                                                                                                               (df['Yearly Replacement Cost'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Working Capital Reserve'] = npv(nominal_IRR, (df['Working Capital Reserve'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Working Capital Reserve'] = (df['Working Capital Reserve'].tolist())[0] + npv(nominal_IRR,
                                                                                                               (df['Working Capital Reserve'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Other Non-Depreciable Capital Cost'] = npv(nominal_IRR, (df['Other Non-Depreciable Capital Cost'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Other Non-Depreciable Capital Cost'] = (df['Other Non-Depreciable Capital Cost'].tolist())[0] + \
                                                                        npv(nominal_IRR, (df['Other Non-Depreciable Capital Cost'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Salvage Value'] = npv(nominal_IRR, (df['Salvage Value'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Salvage Value'] = (df['Salvage Value'].tolist())[0] + npv(nominal_IRR,(df['Salvage Value'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Decommissioning Costs'] = npv(nominal_IRR, (df['Decommissioning Costs'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Decommissioning Costs'] = (df['Decommissioning Costs'].tolist())[0] + npv(nominal_IRR, (df['Decommissioning Costs'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Fixed Operating Cost'] = npv(nominal_IRR, (df['Fixed Operating Cost'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Fixed Operating Cost'] = (df['Fixed Operating Cost'].tolist())[0] + npv(nominal_IRR, (df['Fixed Operating Cost'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Feedstock Cost'] = npv(nominal_IRR, (df['Feedstock Cost'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Feedstock Cost'] = (df['Feedstock Cost'].tolist())[0] + npv(nominal_IRR, (df['Feedstock Cost'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Other Raw Material Cost'] = npv(nominal_IRR, (df['Other Raw Material Cost'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Other Raw Material Cost'] = (df['Other Raw Material Cost'].tolist())[0] + npv(nominal_IRR,(df['Other Raw Material Cost'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Revenue from Byproduct Sales'] = npv(nominal_IRR,(df['Revenue from Byproduct Sales'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Revenue from Byproduct Sales'] = (df['Revenue from Byproduct Sales'].tolist())[0] + npv(nominal_IRR, (df['Revenue from Byproduct Sales'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Other Variable Operating Costs'] = npv(nominal_IRR,(df['Other Variable Operating Costs'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Other Variable Operating Costs'] = (df['Other Variable Operating Costs'].tolist())[0] + npv(nominal_IRR, (df['Other Variable Operating Costs'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Debt Interest'] = npv(nominal_IRR, (df['Debt Interest'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Debt Interest'] = (df['Debt Interest'].tolist())[0] + npv(nominal_IRR, (df['Debt Interest'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Pre-Depreciation Income'] = npv(nominal_IRR, (df['Pre-Depreciation Income'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Pre-Depreciation Income'] = (df['Pre-Depreciation Income'].tolist())[0] + npv(nominal_IRR, (df['Pre-Depreciation Income'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Depreciation Charge'] = npv(nominal_IRR, (df['Depreciation Charge'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Depreciation Charge'] = (df['Depreciation Charge'].tolist())[0] + npv(nominal_IRR, (df['Depreciation Charge'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Taxable Income'] = npv(nominal_IRR, (df['Taxable Income'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Taxable Income'] = (df['Taxable Income'].tolist())[0] + npv(nominal_IRR, (df['Taxable Income'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Total Taxes'] = npv(nominal_IRR, (df['Total Taxes'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Total Taxes'] = (df['Total Taxes'].tolist())[0] + npv(nominal_IRR,
                                                                                       (df['Total Taxes'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'After Income Tax'] = npv(nominal_IRR, (df['After Income Tax'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'After Income Tax'] = (df['After Income Tax'].tolist())[0] + npv(nominal_IRR, (df['After Income Tax'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Principal Payment'] = npv(nominal_IRR, (df['Principal Payment'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Principal Payment'] = (df['Principal Payment'].tolist())[0] + npv(nominal_IRR, (df['Principal Payment'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'After-Tax Post-Depreciation Cash Flow'] = npv(nominal_IRR, (df['After-Tax Post-Depreciation Cash Flow'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'After-Tax Post-Depreciation Cash Flow'] = \
        (df['After-Tax Post-Depreciation Cash Flow'].tolist())[0] + npv(nominal_IRR,
                                                                        (df['After-Tax Post-Depreciation Cash Flow'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Cumulative Cash Flow'] = npv(nominal_IRR, (df['Cumulative Cash Flow'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Cumulative Cash Flow'] = (df['Cumulative Cash Flow'].tolist())[0] + npv(nominal_IRR, (df['Cumulative Cash Flow'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'Pre-Tax Cash Flow'] = npv(nominal_IRR, (df['Pre-Tax Cash Flow'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'Pre-Tax Cash Flow'] = (df['Pre-Tax Cash Flow'].tolist())[0] + npv(nominal_IRR, (df['Pre-Tax Cash Flow'].tolist())[1:useful_life+1])

    df.loc['NPV of Cashflow', 'H2 Sales (kg/year)'] = npv(nominal_IRR, (df['H2 Sales (kg/year)'].tolist())[:useful_life+1])
    df.loc['Discounted Values', 'H2 Sales (kg/year)'] = (df['H2 Sales (kg/year)'].tolist())[0] + npv(after_tax_real_IRR, (df['H2 Sales (kg/year)'].tolist())[1:useful_life+1])

    final_data = pd.DataFrame(
        columns=['Capital Costs', 'Depreciation', 'Principal', 'Operation Costs', 'Tax Incentives', 'H2 Sales (kg)'])
    final_data.loc['Discounted Values', 'Capital Costs'] = -(
            df.loc['Discounted Values', 'Initial Equity Depreciable Capital'] + df.loc[
        'Discounted Values', 'Yearly Replacement Cost'] + df.loc['Discounted Values', 'Working Capital Reserve'] +
            df.loc['Discounted Values', 'Other Non-Depreciable Capital Cost'])
    final_data.loc['Discounted Values', 'Depreciation'] = -df.loc['Discounted Values', 'Depreciation Charge']
    final_data.loc['Discounted Values', 'Principal'] = -df.loc['Discounted Values', 'Principal Payment']
    final_data.loc['Discounted Values', 'Operation Costs'] = -(
            df.loc['Discounted Values', 'Salvage Value'] + df.loc['Discounted Values', 'Decommissioning Costs'] +
            df.loc['Discounted Values', 'Fixed Operating Cost'] + df.loc['Discounted Values', 'Feedstock Cost'] + \
            df.loc['Discounted Values', 'Other Raw Material Cost'] + df.loc[
                'Discounted Values', 'Revenue from Byproduct Sales'] + df.loc[
                'Discounted Values', 'Other Variable Operating Costs'] + df.loc['Discounted Values', 'Debt Interest'])
    final_data.loc['Discounted Values', 'Tax Incentives'] = tax_credit
    final_data.loc['Discounted Values', 'H2 Sales (kg)'] = df.loc['Discounted Values', 'H2 Sales (kg/year)']

    final_data.loc['Tax Coefficient', 'Capital Costs'] = 1
    final_data.loc['LCOH Cost Contribution', 'Capital Costs'] = final_data.loc['Discounted Values', 'Capital Costs'] * \
                                                                final_data.loc['Tax Coefficient', 'Capital Costs']

    final_data.loc['Tax Coefficient', 'Depreciation'] = -total_tax_rate
    final_data.loc['LCOH Cost Contribution', 'Depreciation'] = final_data.loc['Discounted Values', 'Depreciation'] * \
                                                               final_data.loc['Tax Coefficient', 'Depreciation']

    final_data.loc['Tax Coefficient', 'Principal'] = 1
    final_data.loc['LCOH Cost Contribution', 'Principal'] = final_data.loc['Discounted Values', 'Principal'] * \
                                                            final_data.loc['Tax Coefficient', 'Principal']

    final_data.loc['Tax Coefficient', 'Operation Costs'] = 1 - total_tax_rate
    final_data.loc['LCOH Cost Contribution', 'Operation Costs'] = final_data.loc['Discounted Values', 'Operation Costs'] * \
                                                                  final_data.loc['Tax Coefficient', 'Operation Costs']

    final_data.loc['Tax Coefficient', 'Tax Incentives'] = -1
    final_data.loc['LCOH Cost Contribution', 'Tax Incentives'] = final_data.loc['Discounted Values', 'Tax Incentives'] * \
                                                                 final_data.loc['Tax Coefficient', 'Tax Incentives']

    final_data.loc['Tax Coefficient', 'H2 Sales (kg)'] = 1 - total_tax_rate
    final_data.loc['LCOH Cost Contribution', 'H2 Sales (kg)'] = final_data.loc['Discounted Values', 'H2 Sales (kg)'] * \
                                                                final_data.loc['Tax Coefficient', 'H2 Sales (kg)']

    Final_Hydrogen_Cost_Real = ((final_data.loc['LCOH Cost Contribution', 'Capital Costs'] + final_data.loc[
        'LCOH Cost Contribution', 'Depreciation'] + final_data.loc['LCOH Cost Contribution', 'Principal'] + \
                                 final_data.loc['LCOH Cost Contribution', 'Operation Costs'] + final_data.loc[
                                     'LCOH Cost Contribution', 'Tax Incentives']) / final_data.loc[
                                    'LCOH Cost Contribution', 'H2 Sales (kg)']) * (
                                       1 + inflation_rate) ** length_of_construction_period / inflation_factor
    # print(Final_Hydrogen_Cost_Real)

    Cost_Breakdown = pd.DataFrame(columns=['After Tax Present Value', '% of Total', '$/kg of H2'])
    Cost_Breakdown.loc['Capital Related Costs', 'After Tax Present Value'] = -(((df.loc[
                                                                                     'NPV of Cashflow', 'Initial Equity Depreciable Capital'] +
                                                                                 df.loc[
                                                                                     'NPV of Cashflow', 'Yearly Replacement Cost'] +
                                                                                 df.loc[
                                                                                     'NPV of Cashflow', 'Working Capital Reserve'] +
                                                                                 df.loc[
                                                                                     'NPV of Cashflow', 'Other Non-Depreciable Capital Cost']) + \
                                                                                df.loc[
                                                                                    'NPV of Cashflow', 'Salvage Value'] * (
                                                                                        1 - total_tax_rate) + df.loc[
                                                                                    'NPV of Cashflow', 'Debt Interest'] * (
                                                                                        1 - total_tax_rate) + df.loc[
                                                                                    'NPV of Cashflow', 'Depreciation Charge'] * (
                                                                                    -total_tax_rate) + df.loc[
                                                                                    'NPV of Cashflow', 'Principal Payment']) / (
                                                                                       1 - total_tax_rate))
    Cost_Breakdown.loc['Decommissioning Costs', 'After Tax Present Value'] = -df.loc[
        'NPV of Cashflow', 'Decommissioning Costs']
    Cost_Breakdown.loc['Fixed O&M', 'After Tax Present Value'] = -df.loc['NPV of Cashflow', 'Fixed Operating Cost']
    Cost_Breakdown.loc['Feedstock Costs', 'After Tax Present Value'] = df.loc['NPV of Cashflow', 'Feedstock Cost']
    Cost_Breakdown.loc['Other Raw Material Costs', 'After Tax Present Value'] = df.loc[
        'NPV of Cashflow', 'Other Raw Material Cost']
    Cost_Breakdown.loc['Byproduct Credits', 'After Tax Present Value'] = df.loc[
        'NPV of Cashflow', 'Revenue from Byproduct Sales']
    Cost_Breakdown.loc['Other Variable Costs (Utilities)', 'After Tax Present Value'] = -df.loc[
        'NPV of Cashflow', 'Other Variable Operating Costs']

    Cost_Breakdown['% of Total'] = (Cost_Breakdown['After Tax Present Value'] / Cost_Breakdown[
        'After Tax Present Value'].sum()) * 100
    Cost_Breakdown['$/kg of H2'] = (Cost_Breakdown['% of Total'] / 100) * Final_Hydrogen_Cost_Real
    Cost_Breakdown.loc['TOTAL'] = Cost_Breakdown.sum(axis=0)

    # Depreciation_Table.to_csv('Depreciation_Table.csv')
    # df.to_csv('CashFlow.csv')
    # final_data.to_csv('LCOH_Cost_Contribution')
    # Cost_Breakdown.to_csv('Cost_Breakdown.csv')

    # print(Cost_Breakdown)
    feedstock_cost_h2_levelized = Cost_Breakdown.loc['Other Variable Costs (Utilities)', '$/kg of H2']
    results = dict()
    results['Capital Related Costs'] = Cost_Breakdown.loc['Capital Related Costs', '$/kg of H2']
    results['Fixed O&M'] = Cost_Breakdown.loc['Fixed O&M', '$/kg of H2']
    results['Variable Costs/Feedstock'] =Cost_Breakdown.loc['Other Variable Costs (Utilities)', '$/kg of H2']
    results['Total Hydrogen Cost ($/kgH2)'] = Final_Hydrogen_Cost_Real
    results['Total H2A Cost'] = Cost_Breakdown.sum(axis=0)
    results['peak_daily_production_rate'] = peak_daily_production_rate
    results['electrolyzer_size'] = stack_input_power
    results['total_plant_size'] = total_system_input
    results['scaled_total_installed_cost'] = scaled_total_installed_cost
    results['scaled_total_installed_cost_kw'] = scaled_total_installed_cost/(total_system_input*1000)
    results['expenses_annual_cashflow'] = df['After-Tax Post-Depreciation Cash Flow'] - df['Pre-Depreciation Income']

    return results