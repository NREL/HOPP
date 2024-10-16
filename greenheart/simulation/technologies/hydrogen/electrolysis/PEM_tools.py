import numpy as np
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters as PEMClusters,
)

def create_1MW_reference_PEM():
    pem_param_dict = {
        "eol_eff_percent_loss": 10,
        "uptime_hours_until_eol": 77600,
        "include_degradation_penalty": True,
        "turndown_ratio": 0.1,
    }
    pem = PEMClusters(cluster_size_mw=1, plant_life=30, **pem_param_dict)
    return pem


def get_electrolyzer_BOL_efficiency():
    pem_1MW = create_1MW_reference_PEM()
    bol_eff = pem_1MW.output_dict["BOL Efficiency Curve Info"][
        "Efficiency [kWh/kg]"
    ].values[-1]

    return np.round(bol_eff, 2)

def size_electrolyzer_for_hydrogen_demand(
    hydrogen_production_capacity_required_kgphr,
    size_for="BOL",
    electrolyzer_degradation_power_increase=None,
):
    electrolyzer_energy_kWh_per_kg_estimate_BOL = get_electrolyzer_BOL_efficiency()
    if size_for == "BOL":
        electrolyzer_capacity_MW = (
            hydrogen_production_capacity_required_kgphr
            * electrolyzer_energy_kWh_per_kg_estimate_BOL
            / 1000
        )
    elif size_for == "EOL":
        electrolyzer_energy_kWh_per_kg_estimate_EOL = (
            electrolyzer_energy_kWh_per_kg_estimate_BOL
            * (1 + electrolyzer_degradation_power_increase)
        )
        electrolyzer_capacity_MW = (
            hydrogen_production_capacity_required_kgphr
            * electrolyzer_energy_kWh_per_kg_estimate_EOL
            / 1000
        )

    return electrolyzer_capacity_MW


def check_capacity_based_on_clusters(electrolyzer_capacity_BOL_MW, cluster_cap_mw):

    if electrolyzer_capacity_BOL_MW % cluster_cap_mw == 0:
        n_pem_clusters_max = electrolyzer_capacity_BOL_MW // cluster_cap_mw
    else:
        n_pem_clusters_max = int(
            np.ceil(np.ceil(electrolyzer_capacity_BOL_MW) / cluster_cap_mw)
        )
    electrolyzer_size_mw = n_pem_clusters_max * cluster_cap_mw
    return electrolyzer_size_mw