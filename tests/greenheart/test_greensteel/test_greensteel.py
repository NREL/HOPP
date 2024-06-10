import os

from pytest import approx

from greenheart.simulation.greenheart_simulation import (
    run_simulation,
    GreenHeartSimulationConfig,
)

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "input/")


def test_onshore_steel_mn_2030_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_mn_2030.yaml")
    filename_greenheart_config = os.path.join(
        path, "plant/greenheart_config_onshore_mn_2030.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=8,
    )

    output = run_simulation(config)

    with subtests.test("aep"):
        assert output.hopp_results["hybrid_plant"].annual_energies["wind"] == approx(
            4007165638.0913243, rel=0.1
        )

    with subtests.test("wind cf"):
        assert output.hopp_results["hybrid_plant"].capacity_factors["wind"] == approx(
            35.3, rel=0.1
        )

    with subtests.test("h2 capacity factor"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Capacity Factor"
        ] == approx(0.3817629024505595, rel=0.1)

    with subtests.test("electrolyzer installed capex"):
        assert output.capex_breakdown["electrolyzer"] / 1160000 == approx(
            553.4592000000001, rel=0.1
        )

    with subtests.test("lcoh"):
        assert output.lcoh == approx(4.2986685034417045, rel=0.1)

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(71527565.78081538, rel=0.1)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(2603580420.446765, 0.1)

    with subtests.test("wind capex"):
        assert output.capex_breakdown["wind"] == approx(1706186592.0000002, rel=0.1)

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('compressor capex'):
    #     assert output.capex_breakdown['h2_transport_compressor'] == approx(18749265.76357594, rel=0.1)

    with subtests.test("electrolyzer capex"):
        assert output.capex_breakdown["electrolyzer"] == approx(
            642012672.0000001, rel=0.1
        )
    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test("h2 storage capex"):
    #     assert output.capex_breakdown["h2_storage"] == approx(
    #         234630860.95910057, rel=0.1
    #     )

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('desal capex'):
    #     assert output.capex_breakdown['desal'] == approx(294490.93738406134, rel=0.1)

    with subtests.test("lcos"):
        assert output.steel_finance.sol.get("price") == approx(
            961.2866791076059, rel=0.1
        )


def test_onshore_ammonia_tx_2030_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx_2030.yaml")
    filename_greenheart_config = os.path.join(
        path, "plant/greenheart_config_onshore_tx_2030.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=8,
    )

    output = run_simulation(config)

    with subtests.test("lcoh"):
        assert output.lcoh == approx(4.023963541118758, rel=0.1)

    with subtests.test("lcoa"):
        assert output.ammonia_finance.sol.get("price") == approx(
            0.8839797787889466, rel=0.1
        )

    with subtests.test("aep"):
        assert output.hopp_results["hybrid_plant"].annual_energies["wind"] == approx(
            3383439.9656016766 * 1e3, rel=0.1
        )

    with subtests.test("wind cf"):
        assert output.hopp_results["hybrid_plant"].capacity_factors["wind"] == approx(
            36.78, rel=0.1
        )

    with subtests.test("h2 capacity factor"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Capacity Factor"
        ] == approx(0.38749, rel=0.1)

    with subtests.test("electrolyzer installed capex"):
        assert output.capex_breakdown["electrolyzer"] / 960000 == approx(
            553.4592000000001, rel=0.1
        )

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(60400414.51854633, rel=0.1)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(
            1922324028.6785245, rel=0.1
        )

    with subtests.test("wind capex"):
        assert output.capex_breakdown["wind"] == approx(1196393976.0, rel=0.1)

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('compressor capex'):
    #     assert output.capex_breakdown['h2_transport_compressor'] == approx(14082368.770328939, rel=0.1)

    with subtests.test("electrolyzer capex"):
        assert output.capex_breakdown["electrolyzer"] == approx(
            531320832.0000001, rel=0.1
        )
    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test("h2 storage capex"):
    #     assert output.capex_breakdown["h2_storage"] == approx(
    #         178870827.30895007, rel=0.1
    #     )
    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test("desal capex"):
    #     assert output.capex_breakdown["desal"] == approx(1656024.5992454647, rel=0.1)


def test_onshore_ammonia_tx_2030_base_policy(subtests):
    # load inputs as needed
    # NOTE: GreenSteel used Wind PTC, H2 PTC and Storage ITC for base policy and incentive_option 2 does not include the Storage ITC
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx_2030.yaml")
    filename_greenheart_config = os.path.join(
        path, "plant/greenheart_config_onshore_tx_2030.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=2,
        plant_design_scenario=9,
        output_level=8,
    )

    output = run_simulation(config)

    with subtests.test("lcoh"):
        assert output.lcoh == approx(
            3.2231088737405846,
            rel=0.11
        )

    with subtests.test("lcoa"):
        assert output.ammonia_finance.sol.get("price") == approx(
            0.7242460501735473,
             rel=0.1
        )

    with subtests.test("aep"):
        assert output.hopp_results["hybrid_plant"].annual_energies["wind"] == approx(
            3383439.9656016766 * 1e3, 
            rel=0.1
        )

    with subtests.test("wind cf"):
        assert output.hopp_results["hybrid_plant"].capacity_factors["wind"] == approx(
            36.78, rel=0.1
        )

    with subtests.test("h2 capacity factor"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Capacity Factor"
        ] == approx(0.4023306656204429, rel=0.1)

    with subtests.test("electrolyzer installed capex"):
        assert output.capex_breakdown["electrolyzer"] / 960000 == approx(
            553.4592000000001, rel=0.1
        )

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(60400414.51854633, rel=0.1)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(1922324028.6785245, 
                                                              0.1
                                                              )


def test_onshore_ammonia_tx_2030_max_policy(subtests):
    # load inputs as needed
    # NOTE: GreenSteel used Wind PTC, H2 PTC and Storage ITC for max policy and incentive_option 3 does not include the Storage ITC
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx_2030.yaml")
    filename_greenheart_config = os.path.join(
        path, "plant/greenheart_config_onshore_tx_2030.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=7,
        plant_design_scenario=9,
        output_level=8,
    )

    output = run_simulation(config)

    with subtests.test("lcoh"):
        assert output.lcoh == approx(-0.3719231321573331, rel=10)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(1922324028.6785245, 0.1)

    with subtests.test("lcoa"):
        assert output.ammonia_finance.sol.get("price") == approx(
            0.007202259061804786, rel=100
        )

    with subtests.test("aep"):
        assert output.hopp_results["hybrid_plant"].annual_energies["wind"] == approx(
            3383439.9656016766 * 1e3, rel=0.1
        )

    with subtests.test("wind cf"):
        assert output.hopp_results["hybrid_plant"].capacity_factors["wind"] == approx(
            36.78, rel=0.1
        )

    with subtests.test("h2 capacity factor"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Capacity Factor"
        ] == approx(0.3874913813926033, rel=0.1)

    with subtests.test("electrolyzer installed capex"):
        assert output.capex_breakdown["electrolyzer"] / 960000 == approx(
            553.4592000000001, rel=0.1
        )

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(60400414.51854633, rel=0.1)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(1922324028.6785245, 0.1)


def test_onshore_ammonia_tx_2025_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx_2025.yaml")
    filename_greenheart_config = os.path.join(
        path, "plant/greenheart_config_onshore_tx_2025.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=8,
    )

    output = run_simulation(config)

    with subtests.test("lcoh"):
        assert output.lcoh == approx(5.215316325627814, rel=0.1)

    with subtests.test("lcoa"):
        assert output.ammonia_finance.sol.get("price") == approx(
            1.1211844274252674, rel=0.1
        )

    with subtests.test("aep"):
        assert output.hopp_results["hybrid_plant"].annual_energies["wind"] == approx(
            3383439.9656016766 * 1e3, rel=0.1
        )

    with subtests.test("wind cf"):
        assert output.hopp_results["hybrid_plant"].capacity_factors["wind"] == approx(
            36.78, rel=0.1
        )

    with subtests.test("h2 capacity factor"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Capacity Factor"
        ] == approx(0.3874913813926033, rel=0.1)

    with subtests.test("electrolyzer installed capex"):
        assert output.capex_breakdown["electrolyzer"] / 960000 == approx(
            923.3862399999999, rel=0.1
        )

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(60400414.51854633, rel=0.1)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(2400580143.0785246, 0.1)

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(60400414.51854633, rel=0.1)

    with subtests.test("wind capex"):
        assert output.capex_breakdown["wind"] == approx(1319520132.0000002, rel=0.1)

    with subtests.test("electrolyzer capex"):
        assert output.capex_breakdown["electrolyzer"] == approx(
            886450790.3999999, rel=0.1
        )

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test("h2 storage capex"):
    #     assert output.capex_breakdown["h2_storage"] == approx(
    #         178870827.30895007, rel=0.1
    #     )

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('desal capex'):
    #     assert output.capex_breakdown['desal'] == approx(294490.93738406134, rel=0.1)

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('compressor capex'):
    #     assert output.capex_breakdown['h2_transport_compressor'] == approx(18749265.76357594, rel=0.1)

def test_onshore_ammonia_tx_2035_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx_2035.yaml")
    filename_greenheart_config = os.path.join(
        path, "plant/greenheart_config_onshore_tx_2035.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=8,
    )

    output = run_simulation(config)

    with subtests.test("lcoh"):
        assert output.lcoh == approx(3.68491615716891, rel=0.1)

    with subtests.test("lcoa"):
        assert output.ammonia_finance.sol.get("price") == approx(
            0.8161748484435717, rel=0.1
        )

    with subtests.test("aep"):
        assert output.hopp_results["hybrid_plant"].annual_energies["wind"] == approx(
            3383439.9656016766 * 1e3, rel=0.1
        )

    with subtests.test("wind cf"):
        assert output.hopp_results["hybrid_plant"].capacity_factors["wind"] == approx(
            36.78, rel=0.1
        )

    with subtests.test("h2 capacity factor"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Capacity Factor"
        ] == approx(0.3874913813926033, rel=0.1)

    with subtests.test("electrolyzer installed capex"):
        assert output.capex_breakdown["electrolyzer"] / 960000 == approx(
            457.55808, rel=0.1
        )

    with subtests.test("h2 production"):
        assert output.electrolyzer_physics_results["H2_Results"][
            "Life: Annual H2 production [kg/year]"
        ] == approx(60400414.51854633, rel=0.1)

    with subtests.test("capex"):
        assert sum(output.capex_breakdown.values()) == approx(1771419905.4785244, 0.1)

    with subtests.test("wind capex"):
        assert output.capex_breakdown["wind"] == approx(1137554928.0, rel=0.1)

    with subtests.test("electrolyzer capex"):
        assert output.capex_breakdown["electrolyzer"] == approx(439255756.8, rel=0.1)

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test("h2 storage capex"):
    #     assert output.capex_breakdown["h2_storage"] == approx(
    #         178870827.30895007, rel=0.1
    #     )

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('desal capex'):
    #     assert output.capex_breakdown['desal'] == approx(294490.93738406134, rel=0.1)

    # Not included in test suite. Difference between GS and GreenHEART
    # with subtests.test('compressor capex'):
    #     assert output.capex_breakdown['h2_transport_compressor'] == approx(18749265.76357594, rel=0.1)