import PySAM.MhkWave as MhkWave
import PySAM.MhkCosts as MhkCost
import PySAM.Singleowner as Singleowner

from hopp.power_source import *
#TODO: Add dispatch for Wave
# hopp.dispatch.power_sources.wave_dispatch import WaveDispatch

class MHKWavePlant(PowerSource):
    _system_model: MhkWave.MhkWave
    _financial_model: Singleowner.Singleowner
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
        financial_model = None
        cost_model = MhkCost.new()

        super().__init__("MHKWavePlant", site, system_model, financial_model)
        self._system_model.value("wave_resource_data", self.site.wave_resource.data)

        # if 'layout_mode' not in mhk_config.keys():
        #     layout_mode = 'grid'
        # else:
        #     layout_mode = mhk_config['layout_mode']

        if 'device_rating_kw' not in mhk_config.keys():
            raise ValueError("'device_rating_kw' for MHKWavePlant")

        if 'num_devices' not in mhk_config.keys():
            raise ValueError("'num_devices' required for MHKWavePlant")

        self.mhk_wave_rating = mhk_config['device_rating_kw']
        self.num_devices = mhk_config['num_devices']
        self.power_matrix = mhk_config['wave_power_matrix']
        if 'loss_array_spacing' not in mhk_config.keys():
            self._system_model.loss_array_spacing = 0
        else:
            self._system_model.loss_array_spacing = mhk_config['loss_array_spacing']

        @property
        def system_capacity_kw(self):
            return self._system_model.value()