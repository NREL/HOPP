from turtle import hideturtle
import numpy as np

def check_load_met(load, generation_no_storage, generation_with_storage):
    
    #Check if generation (before and after storage) meets this load

    energy_shortfall_no_storage = [x - y for x, y in
                             zip(load,generation_no_storage)]
    energy_shortfall_with_storage = [x - y for x, y in
                             zip(load,generation_with_storage)]
    load_met_no_storage = [1 if x <= 0 else 0 for x in energy_shortfall_no_storage]
    load_met_with_storage = [1 if x <= 0 else 0 for x in energy_shortfall_with_storage]
    perc_time_load_met_no_storage = 100 * np.sum(load_met_no_storage) / len(load_met_no_storage)
    perc_time_load_met_with_storage = 100 * np.sum(load_met_with_storage) / len(load_met_with_storage)

    return perc_time_load_met_no_storage, perc_time_load_met_with_storage, energy_shortfall_no_storage, energy_shortfall_with_storage


if __name__ == "__main__":
    load = [50,50,50]
    generation_no_storage = [40,21,40]
    generation_with_storage = [50,50,50]
    perc_time_load_met_no_storage, perc_time_load_met_with_storage = check_load_met(load,generation_no_storage,generation_with_storage)
    print("Percentage Time Load met with no storage: {} \n Percentage Time Load met with storage: {}".format(perc_time_load_met_no_storage,perc_time_load_met_with_storage))