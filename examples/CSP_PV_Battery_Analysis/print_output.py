from hybrid.hybrid_simulation import HybridSimulation


def print_table_metric(hybrid: HybridSimulation, metric: str, display_name: str=None):

    def value_line(value):
        return "{:>12.2f} | ".format(value)
    
    if display_name is None:
        line = "{:<25} | ".format(metric)
    else:
        line = "{:<25} | ".format(display_name)
        
    line += value_line(hybrid.grid.value(metric))
        
    if hybrid.tower:
        line += value_line(hybrid.tower.value(metric))
    if hybrid.trough:
        line += value_line(hybrid.trough.value(metric))
    if hybrid.pv:
        line += value_line(hybrid.pv.value(metric))
    if hybrid.battery:
        line += value_line(hybrid.battery.value(metric))
    print(line)

def print_BCR_table(hybrid: HybridSimulation):
    # BCR Breakdown
    print("\n ======= Benefit Cost Ratio Breakdown ======= \n")
    header = "{:<25} | {:^12} | ".format('Term', 'Hybrid')

    if hybrid.tower:
        header += "{:^12} | ".format('Tower')
    if hybrid.trough:
        header += "{:^12} | ".format('Trough')
    if hybrid.pv:
        header += "{:^12} | ".format('PV')
    if hybrid.battery:
        header += "{:^12} | ".format('Battery')
    print(header)

    BCR_terms = {"npv_ppa_revenue": "PPA revenue [$]",
                 "npv_capacity_revenue": "Capacity revenue [$]",
                 "npv_curtailment_revenue": "Curtail revenue [$]",
                 "npv_fed_pbi_income": "Federal PBI income [$]",
                 "npv_oth_pbi_income": "Other PBI income [$]",
                 "npv_salvage_value": "Salvage value [$]",
                 "npv_sta_pbi_income": "State PBI income [$]",
                 "npv_uti_pbi_income": "Utility PBI income [$]",
                 "npv_annual_costs": "Annual costs [$]",
                 "benefit_cost_ratio": "Benefit Cost Ratio [-]"}

    for term in BCR_terms.keys():
        print_table_metric(hybrid, term, BCR_terms[term])

def print_tech_output(hybrid: HybridSimulation, tech: str):
    if tech != 'hybrid':
        if not getattr(hybrid, tech):
            return

    print(tech + ":")
    print("\tEnergy (year 1) [GWh]: {:.3f}".format(getattr(hybrid.annual_energies, tech)/1e6))
    print("\tCapacity Factor [%]: {:.2f}".format(getattr(hybrid.capacity_factors, tech)))
    if tech == 'hybrid':
        print("\tInstalled Cost [$M]: {:.2f}".format(getattr(hybrid,'grid').total_installed_cost/1e6))
    else:
        print("\tInstalled Cost [$M]: {:.2f}".format(getattr(hybrid,tech).total_installed_cost/1e6))
    print("\tNPV [$M]: {:.3f}".format(getattr(hybrid.net_present_values, tech)/1e6))
    print("\tLCOE (nominal) [$/MWh]: {:.2f}".format(getattr(hybrid.lcoe_nom, tech)*10))
    print("\tLCOE (real) [$/MWh]: {:.2f}".format(getattr(hybrid.lcoe_real, tech)*10))
    print("\tIRR [%]: {:.2f}".format(getattr(hybrid.internal_rate_of_returns, tech)))
    print("\tBenefit Cost Ratio [-]: {:.2f}".format(getattr(hybrid.benefit_cost_ratios, tech)))
    print("\tCapacity credit [%]: {:.2f}".format(getattr(hybrid.capacity_credit_percent, tech)))
    print("\tCapacity payment (year 1): {:.2f}".format(getattr(hybrid.capacity_payments, tech)[1]))
    if tech == 'hybrid':
        print("\tCurtailment percentage [%]: {:.2f}".format(hybrid.grid.curtailment_percent))

def print_hybrid_output(hybrid: HybridSimulation):
    # TODO: Create a function to print this as a table
    techs = ['tower', 'trough', 'pv', 'battery', 'hybrid']
    for t in techs:
        print_tech_output(hybrid, t)

    if hybrid.site.follow_desired_schedule:
        print("\tMissed load [MWh]: {:.2f}".format(sum(hybrid.grid.missed_load[0:8760])/1.e3))
        print("\tMissed load percentage [%]: {:.2f}".format(hybrid.grid.missed_load_percentage*100.0))
        print("\tSchedule curtailed [MWh]: {:.2f}".format(sum(hybrid.grid.schedule_curtailed[0:8760])/1.e3))
        print("\tSchedule curtailed percentage [%]: {:.2f}".format(hybrid.grid.schedule_curtailed_percentage*100.0))