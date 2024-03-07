from pytest import approx, fixture

from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml
from hopp import ROOT_DIR

@fixture
def hybrid_config():
    """Loads the config YAML and updates site info to use resource files."""
    hybrid_config_path = ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "hybrid_run.yaml"
    hybrid_config = load_yaml(hybrid_config_path)

    return hybrid_config

def test_reinitialize(hybrid_config, subtests):
    """
    Make sure that the interface methods work as expected
    """
    technologies = hybrid_config["technologies"]
    solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind', 'grid')}
    hybrid_config["technologies"] = solar_wind_hybrid
    hi = HoppInterface(hybrid_config)

    hi.simulate()

    with subtests.test("has pv"):
        assert 'pv' in hi.system.technologies.keys()
    with subtests.test("aep with pv"):
        assert hi.system.annual_energies.hybrid == approx(41681662.63, 1e3)

    hybrid_config["technologies"].pop("pv")
    hi.reinitialize(hybrid_config)

    hi.simulate()

    with subtests.test("does not have pv"):
        assert not 'pv' in hi.system.technologies.keys()
    with subtests.test("aep without pv"):
        assert hi.system.annual_energies.hybrid == approx(33615479, 1e3)
