import PySAM.MhkWave as MhkWave
import PySAM.MhkCosts as MhkCost
import PySAM.Singleowner as Singleowner

from hopp.power_source import *
#TODO: Add dispatch for Wave
# hopp.dispatch.power_sources.wave_dispatch import WaveDispatch

class MHKWavePlant(PowerSource):
    _system_model: MhkWave.MhkWave
    #_financial_model: Singleowner.Singleowner
    _cost_model: MhkCost.MhkCosts
    # _layout:
    # _dispatch:

    def __init__(self,
                 site: SiteInfo,
                 mhk_config: dict,
                 ):
        """
        Set up a MhkWavePlant

        :param: dict, with keys ('device_rating_kw', 'num_devices', 'wave_power_matrix',
                    'loss_array_spacing', 'loss_resource_overprediction', 'loss_transmission',
                    'loss_downtime', 'loss_additional', 'layout_mode')
            where losses are optional inputs otherwise default to 0%
            where 'layout_mode' is from MhkGridParameters #TODO: make MhkGridParameters
        """
        system_model = MhkWave.new()
        financial_model = None #Singleowner.from_existing(system_model)
        cost_model = MhkCost.new()

        super().__init__("MHKWavePlant", site, system_model, financial_model)
        #self._system_model.value("wave_resource_data", self.site.wave_resource.data)
        system_model.MHKWave.wave_resource_model_choice = 1 #Time-series data=1 JPD=0
        system_model.MHKWave.significant_wave_height = self.site.wave_resource.data['significant_wave_height'] #significant_wave_height #array, length 2920
        system_model.MHKWave.energy_period = self.site.wave_resource.data['energy_period'] #energy_period #array, length 2920
        system_model.MHKWave.year = self.site.wave_resource.data['year'] # year
        system_model.MHKWave.month = self.site.wave_resource.data['month'] # month
        system_model.MHKWave.day = self.site.wave_resource.data['day'] #day
        system_model.MHKWave.hour = self.site.wave_resource.data['hour'] #hour
        system_model.MHKWave.minute = self.site.wave_resource.data['minute'] #minute
        # if 'layout_mode' not in mhk_config.keys():
        #     layout_mode = 'grid'
        # else:
        #     layout_mode = mhk_config['layout_mode']

        if 'device_rating_kw' not in mhk_config.keys():
            raise ValueError("'device_rating_kw' for MHKWavePlant")

        if 'num_devices' not in mhk_config.keys():
            raise ValueError("'num_devices' required for MHKWavePlant")

        if 'wave_power_matrix' not in mhk_config.keys():
            raise ValueError("'wave_power_matrix' required for MHKWavePlant")

        self.mhk_wave_rating = mhk_config['device_rating_kw']
        self.num_devices = mhk_config['num_devices']
        self.power_matrix = mhk_config['wave_power_matrix']

        self._system_model.MHKWave.device_rated_power = self.mhk_wave_rating
        self._system_model.MHKWave.number_devices = self.num_devices
        self._system_model.MHKWave.wave_power_matrix = self.power_matrix

        # Losses
        if 'loss_array_spacing' not in mhk_config.keys():
            self._system_model.MHKWave.loss_array_spacing = 0
        else:
            self._system_model.MHKWave.loss_array_spacing = mhk_config['loss_array_spacing']
        if 'loss_downtime' not in mhk_config.keys():
            self._system_model.MHKWave.loss_downtime = 0
        else:
            self._system_model.MHKWave.loss_downtime = mhk_config['loss_downtime'] 
        if 'loss_resource_overprediction' not in mhk_config.keys():
            self._system_model.MHKWave.loss_resource_overprediction = 0
        else:
            self._system_model.MHKWave.loss_resource_overprediction = mhk_config['loss_resource_overprediction'] 
        if 'loss_transmission' not in mhk_config.keys():
            self._system_model.MHKWave.loss_transmission = 0
        else:
            self._system_model.MHKWave.loss_transmission = mhk_config['loss_transmission']
        if 'loss_additional' not in mhk_config.keys():
            self._system_model.MHKWave.loss_additional = 0
        else:
            self._system_model.MHKWave.loss_additional = mhk_config['loss_additional']

    @property
    def device_rated_power(self):
        self._system_model.MHKWave.device_rated_power = self.device_rated_power

    @ device_rated_power.setter
    def device_rated_power(self, device_rate_power: float):
        self._system_model.MHKWave.device_rated_power = device_rate_power

    @property
    def wave_power_matrix(self):
        self._system_model.MHKWave.wave_power_matrix = self.power_matrix

    @wave_power_matrix.setter
    def wave_power_matrix(self, wave_power_matrix: dict):
        self._system_model.MHKWave.wave_power_matrix = wave_power_matrix
        #self._system_model.value("wave_power_matrix", wave_power_matrix)

    @property
    def system_capacity_kw(self):
        return self.mhk_wave_rating * self.num_devices

    def system_capacity_by_num_devices(self, wave_size_kw):
        """
        Sets the system capacity by adjusting the number of devices in plant

        :param wave_size_kw: desired system capacity in kW
        """
        new_num_devices = round(wave_size_kw / self.mhk_wave_rating)
        if self.num_devices != new_num_devices:
            self.num_devices = new_num_devices

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity by updates the number of wave devices using device rating
        :param size_kw:
        :return:
        """
        self.system_capacity_by_num_devices(size_kw)

