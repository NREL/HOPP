import pytest
from pytest import fixture
from pathlib import Path

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.wave.mhk_wave_plant import MHKWavePlant, MHKConfig
from hopp.simulation.technologies.financial.mhk_cost_model import MHKCostModelInputs
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.utilities import load_yaml
from hopp import ROOT_DIR
from tests.hopp.utils import DEFAULT_FIN_CONFIG


@fixture
def site():
	data = {
		"lat": 44.6899,
		"lon": 124.1346,
		"year": 2010,
		"tz": -7,
	}
	wave_resource_file = ROOT_DIR / "simulation" / "resource_files" / "wave" / "Wave_resource_timeseries.csv"
	site = SiteInfo(data, solar=False, wind=False, wave=True, wave_resource_file=wave_resource_file)

	return site

@fixture
def mhk_config():
	mhk_yaml_path = Path(__file__).absolute().parent.parent.parent / "tests" / "hopp" / "inputs" / "wave" / "wave_device.yaml"
	mhk_config = load_yaml(mhk_yaml_path)

	return mhk_config

@fixture
def waveplant(mhk_config, site):
	financial_model = {'fin_model': DEFAULT_FIN_CONFIG}
	mhk_config.update(financial_model)
	config = MHKConfig.from_dict(mhk_config)
	
	cost_model_input = MHKCostModelInputs.from_dict({
		'reference_model_num':3,
		'water_depth': 100,
		'distance_to_shore': 80,
		'number_rows': 10,
		'device_spacing':600,
		'row_spacing': 600,
		'cable_system_overbuild': 20
	})
	return MHKWavePlant(site, config, cost_model_input)


def test_mhk_config(mhk_config, subtests):
	with subtests.test("with basic params"):
		financial_model = {'fin_model': DEFAULT_FIN_CONFIG}
		mhk_config.update(financial_model)

		config = MHKConfig.from_dict(mhk_config)

		assert config.device_rating_kw == 286.
		assert config.num_devices == 100
		assert config.wave_power_matrix == mhk_config["wave_power_matrix"]
		assert config.fin_model is not None
		
		# defaults
		assert config.loss_array_spacing == 0.
		assert config.loss_resource_overprediction == 0.
		assert config.loss_transmission == 0.
		assert config.loss_downtime == 0.
		assert config.loss_additional == 0.


def test_system_outputs(waveplant,subtests):
	# Test to see if there have been changes to PySAM MhkWave model and it is able to handle 1-hr 
	# Timeseries data. Right now have to divide hourly data outputs by 3 to get the same values
	waveplant.simulate(25)

	with subtests.test("annual energy kwh"):
		assert waveplant.annual_energy_kwh == pytest.approx(121325260.0,0)

	with subtests.test("capacity factor"):
		assert waveplant.capacity_factor == pytest.approx(48.42,1)

	with subtests.test("number of hours"):
		assert waveplant.numberHours == pytest.approx(8760)


def test_cost_outputs(waveplant):
	waveplant.simulate(25)

	assert waveplant.mhk_costs.cost_outputs['array_cable_system_cost_modeled'] == pytest.approx(13371634.8, abs=1e-6)


def test_changing_n_devices(waveplant, subtests):
	with subtests.test("less devices than rows"):
		with pytest.raises(Exception):
			waveplant.number_devices = 9

	with subtests.test("not grid shape"):
		with pytest.raises(Exception):
			waveplant.number_devices = 11

	with subtests.test("change system capacity"):
		waveplant.number_devices = 50
		assert waveplant.system_capacity_kw == pytest.approx(286*50,0)

	with subtests.test("update cost model - number_devices"):
		waveplant.number_devices = 50
		assert waveplant.mhk_costs.number_devices == waveplant.number_devices

	with subtests.test("update cost model - system_capacity"):
		waveplant.number_devices = 50
		assert waveplant.mhk_costs.system_capacity_kw == waveplant.system_capacity_kw

def test_changing_device_rating(waveplant,subtests):
	with subtests.test("change system capacity"):
		waveplant.device_rated_power = 150
		assert waveplant.system_capacity_kw == waveplant.device_rated_power * waveplant.number_devices
	
	with subtests.test("update cost model - device rated power"):
		waveplant.device_rated_power = 150
		assert waveplant.mhk_costs.device_rated_power == waveplant.device_rated_power
	
	with subtests.test("update cost model - system capacity"):
		waveplant.device_rated_power = 150
		assert waveplant.mhk_costs.system_capacity_kw == waveplant.system_capacity_kw

def test_changing_wave_power_matrix(waveplant):
	waveplant.number_devices = 100
	waveplant.device_rated_power = 360
	waveplant.wave_power_matrix = [
		[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
		[0.25, 0, 0, 0, 0, 4.8, 6.7, 7.9, 9.3, 10.2, 10.1, 9.7, 9, 8.8, 7.6, 7.3, 6.4, 5.6, 5, 4.5, 4, 0],
		[0.75, 0, 0, 0, 0, 12.3, 16.5, 18.8, 21.2, 22.9, 22.2, 20.9, 19.4, 18.7, 16.5, 16, 14.2, 12.8, 11.5, 10.4, 9.4, 0],
		[1.25, 0, 0, 0, 0, 31.8, 40.7, 44.6, 48.5, 51.7, 48.8, 45.1, 41.8, 40.1, 36.2, 35.1, 31.9, 29.2, 26.5, 24.3, 22, 0],
		[1.75, 0, 0, 0, 0, 58.3, 72.3, 77.1, 81.7, 86.5, 80.8, 74, 69.7, 66.7, 59.7, 57.6, 52.7, 48.7, 44.5, 41.1, 37.6, 0],
		[2.25, 0, 0, 0, 0, 91.3, 110.4, 115.7, 119.3, 126.5, 117.3, 107.9, 102, 97.1, 86.4, 82.6, 75.6, 70.5, 64.7, 60.3, 55.3, 0],
		[2.75, 0, 0, 0, 0, 130.5, 154.9, 160, 162.7, 171.7, 158.5, 145.4, 137.5, 130.4, 115.6, 109.7, 101.4, 94.6, 86.6, 80.8, 74, 0],
		[3.25, 0, 0, 0, 0, 174.9, 204.4, 208.9, 210.4, 220.5, 202.7, 185.4, 175.4, 165.9, 148, 140.3, 129.7, 120.5, 110.1, 102.2, 93.4, 0],
		[3.75, 0, 0, 0, 0, 223.9, 258.5, 261.9, 261.6, 272.4, 249.5, 227.7, 215.3, 204.5, 183.2, 173, 159.8, 147.9, 134.8, 124.8, 113.7, 0],
		[4.25, 0, 0, 0, 0, 277.2, 316.8, 318.5, 316, 327, 298.4, 271.6, 257.2, 245.5, 220.2, 207.3, 191.5, 177.1, 161.8, 149.7, 136.8, 0],
		[4.75, 0, 0, 0, 0, 334.5, 360, 360, 360, 360, 349.4, 317.2, 302.2, 288.2, 258.7, 243.1, 225.4, 208.6, 190.3, 176.1, 160.7, 0],
		[5.25, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 348.9, 332.4, 298.6, 280.1, 261.3, 241.4, 220, 203.3, 185.2, 0],
		[5.75, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 339.7, 319.1, 298.4, 275.5, 250.8, 231.5, 210.7, 0],
		[6.25, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 340.8, 314.3, 285.8, 263.5, 239.7, 0],
		[6.75, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 358.6, 325.8, 300.1, 272.6, 0],
		[7.25, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 341.6, 310.1, 0],
		[7.75, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 352.8, 0],
		[8.25, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 0],
		[8.75, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 0],
		[9.25, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 0],
		[9.75, 0, 0, 0, 0, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 0]
	]
	waveplant.simulate(25)
	assert waveplant.annual_energy_kwh == pytest.approx(160673260.0,0)

def test_changing_system_capacity(waveplant,subtests):
	system_size_kw = 20000

	with subtests.test("system capacity"):
		waveplant.system_capacity_by_num_devices(system_size_kw)
		assert waveplant.system_capacity_kw == waveplant.device_rated_power * round(system_size_kw/waveplant.device_rated_power)
	
	with subtests.test("cost model - system capacity"):
		waveplant.system_capacity_by_num_devices(system_size_kw)
		assert waveplant.mhk_costs.system_capacity_kw == waveplant.device_rated_power * round(system_size_kw/waveplant.device_rated_power)

def test_changing_ref_model(waveplant,subtests):
	waveplant.device_rated_power = 360
	waveplant.number_devices = 100
	waveplant.mhk_costs.ref_model_num = 5
	waveplant.mhk_costs.simulate_costs()

	with subtests.test("model number"):
		assert waveplant.mhk_costs._cost_model.value("lib_wave_device") == "RM5"
	
	with subtests.test("cost model"):
		assert waveplant.mhk_costs.cost_outputs['array_cable_system_cost_modeled'] == pytest.approx(13570902.0, 0)

	with subtests.test("ref model number wrong"):
		with pytest.raises(Exception):
			waveplant.mhk_costs.ref_model_num = 11