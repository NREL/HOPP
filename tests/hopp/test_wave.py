import pytest
from pathlib import Path
import yaml
from yamlinclude import YamlIncludeConstructor

from hopp.sites import SiteInfo
from hopp.mhk_wave_source import MHKWavePlant

data = {
    "lat": 44.6899,
    "lon": 124.1346,
    "year": 2010,
    "tz": -7,
    'no_solar': "True",
    'no_wind': "True"
}

wave_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wave" / "Wave_resource_timeseries.csv"
site = SiteInfo(data, wave_resource_file=wave_resource_file)

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=Path(__file__).absolute())
mhk_yaml_path = "input/wave/wave_device.yaml"
with open(mhk_yaml_path, 'r') as stream:
    mhk_config = yaml.safe_load(stream)

cost_model_inputs = {
	'reference_model_num':3,
	'water_depth': 100,
	'distance_to_shore': 80,
    'number_rows': 10,
	'devices_per_row':10,
	'device_spacing':600,
	'row_spacing': 600,
	'cable_system_overbuild': 20
}

def test_changing_n_devices():
    # test with gridded layout
    model = MHKWavePlant(site = site, mhk_config=mhk_config,cost_model_inputs=cost_model_inputs)
    assert(model.system_capacity_kw == 28600)
    for n in range(1, 20):
        model.number_devices = n
        assert model.number_devices == n, "n turbs should be " + str(n)
        assert model.system_capacity_kw == pytest.approx(28600, 1), "system capacity different when n turbs " + str(n)

def test_changing_device_rating():
    # powercurve scaling
    model = MHKWavePlant(site = site, mhk_config=mhk_config,cost_model_inputs=cost_model_inputs)
    n_devices = model.number_devices
    for n in range(1000, 3000, 150):
        model.device_rated_power = n
        assert model.system_capacity_kw == model.device_rated_power * n_devices, "system size error when rating is " + str(n)

def test_changing_wave_power_matrix():
	model = MHKWavePlant(site = site, mhk_config=mhk_config,cost_model_inputs=cost_model_inputs)
	model.power_matrix = [
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
	model.simulate(25)
	assert model.annual_energy_kw == pytest.approx(76319798.5,1)

def test_changing_system_capacity():
    # adjust number of devices, system capacity won't be exactly as requested
    model = MHKWavePlant(site, mhk_config,cost_model_inputs=cost_model_inputs)
    rating = model.device_rated_power
    for n in range(1000, 20000, 1000):
        model.system_capacity_by_num_devices(n)
        assert model.device_rated_power == rating, str(n)
        assert model.system_capacity_kw == rating * round(n/rating)

def test_system_outputs():
	# Test to see if there have been changes to PySAM MhkWave model and it is able to handle 1-hr 
	# Timeseries data. Right now have to divide hourly data outputs by 3 to get the same values
	model = MHKWavePlant(site, mhk_config,cost_model_inputs=cost_model_inputs)
	model.simulate(25)

	assert model.annual_energy_kw == pytest.approx(121325260.0)
	assert model.capacity_factor == pytest.approx(48.42,1)
	assert model.numberHours == pytest.approx(8760)
