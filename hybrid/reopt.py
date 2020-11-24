import json
import os
import pandas as pd
import requests
import time

from hybrid.solar_source import *
from hybrid.wind_source import WindPlant
from hybrid.storage import Battery
from hybrid.log import hybrid_logger as logger
from hybrid.keys import get_developer_nrel_gov_key
from hybrid.utility_rate import UtilityRate

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class REopt:
    """
    Class to interact with REopt API
    https://developer.nrel.gov/docs/energy-optimization/reopt-v1/
    Presently limited to take a annual wholesale rate to provide system compensation

    https://internal-apis.nrel.gov/reopt-dev/v1/?format=json
    Internal development API that has newer features
    """
    def __init__(self, lat, lon,
                 interconnection_limit_kw: float,
                 load_profile: Sequence,
                 urdb_label: str,
                 solar_model: SolarPlant = None,
                 wind_model: WindPlant = None,
                 storage_model: Battery = None,
                 fin_model: Singleowner = None,
                 fileout=None):
        """
        Initialize REopt API call

        Parameters
        ---------
        lat: float
            The latitude
        lon: float
            The longitude
        wholesale_rate_above_site_load_us_dollars_per_kwh: float
            Price of electricity sold back to the grid above the site load, regardless of net metering
        interconnection_limit_kw: float
            The limit on system capacity size that can be interconnected to grid
        load_profile: list
            The kW demand of the site at every hour (length 8760)
        urdb_label: string
            The identifier of a urdb_rate to use
        *_model: PySAM models
            Models initialized with correct parameters
        fileout: string
            Filename where REopt results should be written
        """

        self.latitude = lat
        self.longitude = lon

        self.interconnection_limit_kw = interconnection_limit_kw
        self.urdb_label = urdb_label
        self.load_profile = load_profile
        self.results = None

        # paths
        self.path_current = os.path.dirname(os.path.abspath(__file__))
        self.path_results = os.path.join(self.path_current, '..', 'results')
        self.path_rates = os.path.join(self.path_current, '..', 'resource_files', 'utility_rates')
        if not os.path.exists(self.path_rates):
            os.makedirs(self.path_rates)
        self.fileout = os.path.join(self.path_results, 'REoptResults.json')

        if fileout is not None:
            self.fileout = fileout

        self.reopt_api_post_url = 'https://developer.nrel.gov/api/reopt/v1/job?format=json'
        self.reopt_api_poll_url = 'https://developer.nrel.gov/api/reopt/v1/job/'

        self.post = self.create_post(solar_model, wind_model, storage_model, fin_model)
        logger.info("Created REopt post")

    def set_rate_path(self, path_rates):
        self.path_rates = path_rates

    @staticmethod
    def PV(solar_model: SolarPlant):
        """ The PV dictionary required by REopt"""

        PV = None
        if solar_model is not None:
            perf_model = solar_model.system_model
            PV = dict()

            if isinstance(perf_model, Pvsam.Pvsamv1):
                track_mode = perf_model.SystemDesign.subarray1_track_mode
                en_backtrack = perf_model.SystemDesign.subarray1_backtrack

                pvsam_to_pvwatts_track = dict()
                pvsam_to_pvwatts_track[0] = 0  # fixed ground mount
                pvsam_to_pvwatts_track[1] = 2  # 1-axis tracker
                if en_backtrack:
                    pvsam_to_pvwatts_track[1] = 3  # 1-axis with backtrack
                pvsam_to_pvwatts_track[2] = 4  # 2-axis tracker
                PV['array_type'] = pvsam_to_pvwatts_track[track_mode]
                PV['tilt'] = perf_model.SystemDesign.subarray1_tilt
                PV['gcr'] = perf_model.SystemDesign.subarray1_gcr
                PV['azimuth'] = perf_model.SystemDesign.subarray1_azimuth
                PV['module_type'] = 1
                inv_type = int(perf_model.Inverter.inverter_model)
                inv_eff_names = ("inv_snl_eff_cec", "inv_ds_eff", "inv_pd_eff", "inv_cec_cg_eff")
                PV['inv_eff'] = perf_model.Inverter.__getattribute__(inv_eff_names[inv_type]) / 100
                inv_ac_names = ("inv_snl_paco", "inv_ds_paco", "inv_pd_paco", "inv_cec_cg_paco")
                PV['dc_ac_ratio'] = perf_model.SystemDesign.system_capacity_kw * 1000 / \
                                    (perf_model.Inverter.__getattribute__(inv_ac_names[inv_type]) * perf_model.Inverter.inverter_count)
            else:
                PV['array_type'] = perf_model.SystemDesign.array_type
                PV['tilt'] = perf_model.SystemDesign.tilt
                PV['gcr'] = perf_model.SystemDesign.gcr
                PV['azimuth'] = perf_model.SystemDesign.azimuth
                PV['inv_eff'] = perf_model.SystemDesign.inv_eff / 100
                PV['dc_ac_ratio'] = perf_model.SystemDesign.dc_ac_ratio
                PV['module_type'] = perf_model.SystemDesign.module_type

            PV['radius'] = 200

            fin_model: Singleowner.Singleowner = solar_model.financial_model
            if fin_model is not None:
                PV['federal_itc_pct'] = fin_model.TaxCreditIncentives.itc_fed_percent * 0.01
                PV['om_cost_us_dollars_per_kw'] = fin_model.SystemCosts.om_capacity[0]
        return PV

    @staticmethod
    def Wind(wind_model: WindPlant):
        """ The Wind dictionary required by REopt"""

        Wind = None
        if wind_model is not None:
            perf_model = wind_model.system_model
            Wind = dict()
            resource_file = wind_model.site.wind_resource.filename
            if os.path.exists(resource_file):
                df_wind = pd.read_csv(resource_file, skiprows=5,
                                      names=['Temperature (C)', 'Pressure (atm)', 'Speed (m/s)', 'Direction (deg)'])

                # Truncate any leap day effects by shaving off December 31
                Wind['temperature_celsius'] = df_wind['Temperature (C)'].tolist()[0:8760]
                Wind['pressure_atmospheres'] = df_wind['Pressure (atm)'].tolist()[0:8760]
                Wind['wind_meters_per_sec'] = df_wind['Speed (m/s)'].tolist()[0:8760]
                Wind['wind_direction_degrees'] = df_wind['Direction (deg)'].tolist()[0:8760]

        fin_model = wind_model.financial_model
        if fin_model is not None:
            Wind['federal_itc_pct'] = 0
            Wind['pbi_us_dollars_per_kwh'] = fin_model.TaxCreditIncentives.ptc_fed_amount[0]
            Wind['pbi_years'] = fin_model.TaxCreditIncentives.ptc_fed_term
            Wind['size_class'] = 'large'
            Wind['installed_cost_us_dollars_per_kw'] = fin_model.SystemCosts.total_installed_cost / \
                                                       fin_model.FinancialParameters.system_capacity
            Wind['om_cost_us_dollars_per_kw'] = fin_model.SystemCosts.om_capacity[0]

        return Wind

    @staticmethod
    def Storage(storage_model: Battery):
        # TODO:
        return None

    @staticmethod
    def financial(hybrid_fin_model: Singleowner.Singleowner):
        """ The financial dictionary required by REopt"""

        financial_dict = dict()
        financial_dict['analysis_years'] = hybrid_fin_model.FinancialParameters.analysis_period
        financial_dict['offtaker_discount_pct'] = \
            (1 + hybrid_fin_model.FinancialParameters.real_discount_rate * 0.01) * \
            (1 + hybrid_fin_model.FinancialParameters.inflation_rate * 0.01) - 1
        financial_dict['offtaker_tax_pct'] = (hybrid_fin_model.FinancialParameters.state_tax_rate[0] * 0.01+
                                              hybrid_fin_model.FinancialParameters.federal_tax_rate[0]) * 0.01
        om_escal_nom = (1 + hybrid_fin_model.SystemCosts.om_capacity_escal * 0.01) * \
                       (1 + hybrid_fin_model.FinancialParameters.inflation_rate * 0.01) - 1
        financial_dict['om_cost_escalation_pct'] = om_escal_nom
        financial_dict['escalation_pct'] = hybrid_fin_model.Revenue.ppa_escalation * 0.01
        return financial_dict

    def tariff(self, fin_model: Singleowner.Singleowner):
        """The ElectricityTariff dict required by REopt."""

        tariff_dict = dict()

        ur = UtilityRate(path_rates=self.path_rates, urdb_label=self.urdb_label)
        if ur is not None:
            tariff_dict['urdb_response'] = ur.get_urdb_response()
        else:
            tariff_dict['urdb_label'] = self.urdb_label

        # currently assume a Single Owner Merchant plant not in a NEM scenario
        tariff_dict['wholesale_rate_above_site_load_us_dollars_per_kwh'] = fin_model.Revenue.ppa_price_input[0]
        tariff_dict['wholesale_rate_us_dollars_per_kwh'] = tariff_dict['wholesale_rate_above_site_load_us_dollars_per_kwh']

        if self.interconnection_limit_kw is not None:
            tariff_dict['interconnection_limit_kw'] = self.interconnection_limit_kw

        return tariff_dict

    def create_post(self, solar_model: SolarPlant, wind_model: WindPlant, batt_model: Battery, hybrid_fin: Singleowner):
        """ The HTTP POST required by REopt"""

        post = dict()

        post['Scenario'] = dict({'user_id': 'hybrid_systems'})
        post['Scenario']['Site'] = dict({'latitude': self.latitude, 'longitude': self.longitude})
        post['Scenario']['Site']['ElectricTariff'] = self.tariff(hybrid_fin)

        if self.load_profile is None:
            self.load_profile = 8760 * [0.0]
        post['Scenario']['Site']['LoadProfile'] = dict({'loads_kw': self.load_profile})

        post['Scenario']['Site']['Financial'] = self.financial(hybrid_fin)

        post['Scenario']['Site']['PV'] = self.PV(solar_model)
        post['Scenario']['Site']['PV']['max_kw'] = self.interconnection_limit_kw

        post['Scenario']['Site']['Wind'] = self.Wind(wind_model)
        post['Scenario']['Site']['Wind']['max_kw'] = self.interconnection_limit_kw

        if batt_model is not None:
            post['Scenario']['Site']['Storage'] = self.Storage(batt_model)

        # write file to results for debugging
        post_path = os.path.join(self.path_results, 'post.json')
        with open(post_path, 'w') as outfile:
            json.dump(post, outfile)

        logger.info("Created REopt post, exported to " + post_path)
        return post

    def get_reopt_results(self, force_download=False):
        """
        Call the reopt_api

        Parameters
        ---------
        force_download: bool
           Whether to force a new api call if the results file already is on disk

        Returns
        ------
        results: dict
            A dictionary of REopt results, as defined
        """

        logger.info("REopt getting results")
        results = dict()
        success = os.path.isfile(self.fileout)
        if not success or force_download:
            post_url = self.reopt_api_post_url + '&api_key={api_key}'.format(api_key=get_developer_nrel_gov_key())
            resp = requests.post(post_url, json.dumps(self.post), verify=False)

            if resp.ok:
                run_id_dict = json.loads(resp.text)

                try:
                    run_id = run_id_dict['run_uuid']
                except KeyError:
                    msg = "Response from {} did not contain run_uuid.".format(post_url)
                    raise KeyError(msg)

                poll_url = self.reopt_api_poll_url + '{run_uuid}/results/?api_key={api_key}'.format(
                    run_uuid=run_id,
                    api_key=get_developer_nrel_gov_key())
                results = self.poller(url=poll_url)
                with open(self.fileout, 'w') as fp:
                    json.dump(obj=results, fp=fp)
                logger.info("Received REopt response, exported to {}".format(self.fileout))
            else:
                text = json.loads(resp.text)
                if "messages" in text.keys():
                    logger.error("REopt response reading error: " + str(text['messages']))
                    raise Exception(text["messages"])
                resp.raise_for_status()
        elif success:
            with open(self.fileout, 'r') as fp:
                results = json.load(fp=fp)
            logger.info("Read REopt response from {}".format(self.fileout))

        return results


    @staticmethod
    def poller(url, poll_interval=2):
        """
        Function for polling the REopt API results URL until status is not "Optimizing..."

        Parameters
        ----------
        url: string
            results url to poll
        poll_interval: float
            seconds

        Returns
        -------
        response: dict
            The dictionary response from the API (once status is not "Optimizing...")
        """
        key_error_count = 0
        key_error_threshold = 4
        status = "Optimizing..."
        while True:

            resp = requests.get(url=url, verify=False)
            resp_dict = json.loads(resp.text)

            try:
                status = resp_dict['outputs']['Scenario']['status']
            except KeyError:
                key_error_count += 1
                if key_error_count > key_error_threshold:
                    break

            if status != "Optimizing...":
                break
            else:
                time.sleep(poll_interval)

        return resp_dict