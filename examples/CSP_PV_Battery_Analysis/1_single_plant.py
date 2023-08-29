from examples.CSP_PV_Battery_Analysis.print_output import print_BCR_table, print_hybrid_output
from examples.CSP_PV_Battery_Analysis.simulation_init import init_hybrid_plant, get_example_path_root

if __name__ == '__main__':
    print_summary_results = True

    # Cases to run with technologies to include
    cases = {
        'pv_batt': ['pv', 'battery'],
        'tower': ['tower'],
        'tower_pv': ['tower', 'pv'],
        'tower_pv_batt': ['tower', 'pv', 'battery'],
        'trough': ['trough'],
        'trough_pv': ['trough', 'pv'],
        'trough_pv_batt': ['trough', 'pv', 'battery']
        }

    for c in cases.keys():
        hybrid = init_hybrid_plant(cases[c], is_test = True)
        hybrid.simulate()
        hybrid.hybrid_simulation_outputs(filename= get_example_path_root() + c + "_outputs.csv")

        # Printing out results
        if print_summary_results:
            print_hybrid_output(hybrid)
            print_BCR_table(hybrid)