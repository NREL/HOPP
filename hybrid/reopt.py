import json
import os
import pandas as pd
import requests
import time
import math

from keys import developer_nrel_gov_key
from hybrid.utility_rate import UtilityRate

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run_reopt(lat,
              lon,
              wholesale_rate_dollar_per_kwh=0,
              net_metering_limit_kw=0,
              systems=None,
              defaults=None,
              reopt_constraints=None,
              force_download=False,
              fileout=None,
              update_scenario=False):
    """
    Run a REopt optimization to get the initial system size at different wholesale rates ($/kWh)

    Function to take in a default dictionary and site dictionary, and update the path names in the default dictionary
    and possibly download the resource data if not present.

    Parameters:
    -----------
    lat : float
        Site latitude
    lon: float
        Site longitude
    wholesale_rate_dollar_per_kwh: float
        The wholesale rate for energy above the site load
    net_metering_limit_kw: float
        The power limit for which net metering is allowed
    systems: dict
        Dictionary of current systems being evaluated
    defaults: dict
        Dictionary of defaults by technology
    reopt_constraints: dict
        Dictionary of additional constraints to be passed to REopt
    force_download: bool
        Boolean flag on whether to force a new API call if results already exist
    fileout: string
        File path to write results to
    update_scenario: bool
        Boolean flag on whether to update propoerties in current scenario

    Returns:
    --------
    systems: dict
        updated systems
    defaults: dict
        updated defaults
    """
    print("\n\nRunning REopt:\n=========================\n")

    wholesale_rate_above_site_load_dollar_per_kwh = defaults["Generic"]["Singleowner"]["PPAPrice"]["ppa_price_input"],
    interconnection_limit = 20000 # defaults["Grid"]["Grid"]["grid_interconnection_limit_kwac"]

    if fileout is None:
        fileout = os.path.join('results',
                               'reopt_results_' +
                               str(lat) + '_' +
                               str(lon) +
                               '_rate_' + str(round(wholesale_rate_above_site_load_dollar_per_kwh[0], 2)) +
                               '_ic_' + str(interconnection_limit) + 'kw.json')

    reopt = REopt(lat=lat,
                  lon=lon,
                  wholesale_rate_dollar_per_kwh=wholesale_rate_dollar_per_kwh,
                  wholesale_rate_above_site_load_us_dollars_per_kwh=wholesale_rate_above_site_load_dollar_per_kwh[0],
                  interconnection_limit_kw=interconnection_limit,
                  net_metering_limit_kw=net_metering_limit_kw,
                  tech_defaults=defaults,
                  reopt_constraints=reopt_constraints,
                  fileout=fileout)
    results = reopt.get_reopt_results(force_download=force_download)

    # Initialize optimization results with REopt sizes
    wind_size_kw = results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"]

    # update scenario to include REopt results
    if update_scenario:
        if wind_size_kw == 0 or wind_size_kw is None:
            defaults.pop("Wind")
        else:
            defaults["Wind"]["Windpower"]["Farm"]["system_capacity"] = wind_size_kw
            turbine_output_kw = max(defaults["Wind"]["Windpower"]["Turbine"]["wind_turbine_powercurve_powerout"])
            n_turbines = math.ceil(wind_size_kw / turbine_output_kw)
            xCoords = defaults["Wind"]["Windpower"]["Farm"]['wind_farm_xCoordinates'][0:n_turbines]
            yCoords = defaults["Wind"]["Windpower"]["Farm"]['wind_farm_yCoordinates'][0:n_turbines]
            defaults["Wind"]["Windpower"]["Farm"]['wind_farm_xCoordinates'] = xCoords
            defaults["Wind"]["Windpower"]["Farm"]['wind_farm_yCoordinates'] = yCoords

        # For solar, need to extract module power rating so can calculate scale number of strings. For now scale by 20 MW default
        nstrings = defaults["Solar"]["Pvsamv1"]["SystemDesign"]["subarray1_nstrings"]
        nstrings = math.ceil(nstrings * results["outputs"]["Scenario"]["Site"]["PV"]["size_kw"] / 20000)
        if nstrings < 10:
            defaults.pop("Solar")
        else:
            ninverters = defaults["Solar"]["Pvsamv1"]["SystemDesign"]["inverter_count"]
            ninverters = math.ceil(ninverters * results["outputs"]["Scenario"]["Site"]["PV"]["size_kw"] / 20000)
            defaults["Solar"]["Pvsamv1"]["SystemDesign"]["inverter_count"] = max(ninverters, 1)
            defaults["Solar"]["Pvsamv1"]["SystemDesign"]["subarray1_nstrings"] = max(nstrings, 1)

        # Update simulation to remove technologies not chosen by REopt (may want to reapproach)
        if 'Solar' not in defaults and 'Solar' in systems:
            systems.pop('Solar')
        if 'Wind' not in defaults and 'Wind' in systems:
            systems.pop('Wind')

    print('---------REopt Results-----------------')
    print('REopt status: ' + results['outputs']['Scenario']['status'])
    print('REopt PV size: ' + str(results['outputs']['Scenario']['Site']['PV']['size_kw']))
    print('REopt Wind size: ' + str(results['outputs']['Scenario']['Site']['Wind']['size_kw']))
    print('REopt Storage power: ' + str(results['outputs']['Scenario']['Site']['Storage']['size_kw']))
    print('REopt Storage energy: ' + str(results['outputs']['Scenario']['Site']['Storage']['size_kwh']))
    print('REopt NPV: ' + str(round(results['outputs']['Scenario']['Site']['Financial']['npv_us_dollars']/1000000, 0)) + ' (million)')
    print('Wholesale rate (up to site load): ' + str(wholesale_rate_dollar_per_kwh))
    print('Wholesale rate (above site load): ' + str(wholesale_rate_above_site_load_dollar_per_kwh))
    print('--------------------------------------------------------')
    return systems, defaults


class REopt:
    """
    Class to interact with REopt API
    https://developer.nrel.gov/docs/energy-optimization/reopt-v1/
    Presently limited to take a annual wholesale rate to provide system compensation

    https://internal-apis.nrel.gov/reopt-dev/v1/?format=json
    Internal development API that has newer features
    """
    def __init__(self, lat, lon,
                 net_metering_limit_kw=0,
                 wholesale_rate_dollar_per_kwh=0,
                 wholesale_rate_above_site_load_us_dollars_per_kwh=0,
                 interconnection_limit_kw=None,
                 load_profile=None,
                 urdb_label=None,
                 tech_defaults=None,
                 reopt_contraints=None,
                 fileout=None,
                 dev_api=True,
                 **kwargs):
        """
        Initialize REopt API call

        Parameters
        ---------
        lat: float
            The latitude
        lon: float
            The longitude
        wholesale_rate: float
            The $/kWh that generation is compensated at above the net metering limit, but below the site load
        wholesale_rate_above_site_load_us_dollars_per_kwh: float
            The $/kWh that generation is compensated at above the net metering limit, but below the site load
        interconnection_limit_kw: float
            The limit on system capacity size that can be interconnected to grid
        load_profile: list
            The kW demand of the site at every hour (length 8760)
        urdb_label: string
            The identifier of a urdb_rate to use
        tech_defaults: dict
            The Scenario defaults which will populate REopt inputs
        reopt_contraints: dict
            Additional constraints for REopt to use on a case-by-case basis
        fileout: string
            Filename where REopt results should be written
        dev_api: bool
            Boolean on whether to use the development API for REopt or not
        """

        self.latitude = lat
        self.longitude = lon
        self.net_metering_limit_kw = net_metering_limit_kw
        self.wholesale_rate = wholesale_rate_dollar_per_kwh

        self.wholesale_rate_above_site_load_us_dollars_per_kwh = wholesale_rate_above_site_load_us_dollars_per_kwh
        self.interconnection_limit_kw = interconnection_limit_kw
        self.urdb_label = urdb_label
        self.load_profile = load_profile
        self.api_key = developer_nrel_gov_key
        self.tech_defaults = tech_defaults
        self.results = None

        # paths
        self.path_current = os.path.dirname(os.path.abspath(__file__))
        self.path_results = os.path.join(self.path_current, '..', 'results')
        self.path_rates = os.path.join(self.path_current, '..', 'resource_files', 'utility_rates')
        self.fileout = os.path.join(self.path_results, 'REoptResults.json')

        if fileout is not None:
            self.fileout = fileout


        # API endpoint
        self.reopt_api_post_url = 'https://developer.nrel.gov/api/reopt/v1/job?format=json'
        self.reopt_api_poll_url = 'https://developer.nrel.gov/api/reopt/v1/job/'
        if dev_api:
            self.reopt_api_post_url = 'https://reopt-dev-api1.nrel.gov/v1/job/?format=json'
            self.reopt_api_poll_url = 'https://reopt-dev-api1.nrel.gov/v1/job/'

        # update any passed in
        self.__dict__.update(kwargs)

    def set_rate_path(self, path_rates):
        self.path_rates = path_rates

    @property
    def PV(self):
        """ The PV dictionary required by REopt"""

        PV = None
        if 'Solar'in self.tech_defaults:
            solar = self.tech_defaults['Solar']['Pvsamv1']
            singleowner = self.tech_defaults['Solar']['Singleowner']
            track_mode = solar['SystemDesign']['subarray1_track_mode']
            en_backtrack = solar['SystemDesign']['subarray1_backtrack']

            pvsam_to_pvwatts_track = dict()
            pvsam_to_pvwatts_track[0] = 0  # fixed ground mount
            pvsam_to_pvwatts_track[1] = 2  # 1-axis tracker
            if en_backtrack:
                pvsam_to_pvwatts_track[1] = 3  # 1-axis with backtrack
            pvsam_to_pvwatts_track[2] = 4  # 2-axis tracker

            # REopt API inputs
            PV = dict()
            PV['array_type'] = pvsam_to_pvwatts_track[track_mode]
            PV['azimuth'] = solar['SystemDesign']['subarray1_azimuth']
            PV['dc_ac_ratio'] = solar['SystemDesign']['system_capacity'] * 1000 / \
                                (solar['InverterCECDatabase']['inv_snl_paco'] * solar['Inverter']['inverter_count'])
            PV['degradation_pct'] = solar['Lifetime']['dc_degradation'][0]
            PV['gcr'] = solar['SystemDesign']['subarray1_gcr']
            PV['radius'] = 200
            PV['tilt'] = solar['SystemDesign']['subarray1_tilt']
            PV['federal_itc_pct'] = singleowner['TaxCreditIncentives']['itc_fed_percent']*0.01
            PV['installed_cost_us_dollars_per_kw'] = singleowner['SystemCosts']['total_installed_cost'] / \
                                                     singleowner['FinancialParameters']['system_capacity']
            PV['om_cost_us_dollars_per_kw'] = singleowner['SystemCosts']['om_capacity'][0]

        return PV

    @property
    def Wind(self):
        """ The Wind dictionary required by REopt"""

        Wind = None
        if 'Wind' in self.tech_defaults:
            singleowner = self.tech_defaults['Wind']['Singleowner']

            # REopt API inputs
            Wind = dict()
            Wind['federal_itc_pct'] = 0
            Wind['pbi_us_dollars_per_kwh'] = singleowner['TaxCreditIncentives']['ptc_fed_amount'][0]
            Wind['pbi_years'] = singleowner['TaxCreditIncentives']['ptc_fed_term']
            Wind['size_class'] = 'large'
            Wind['installed_cost_us_dollars_per_kw'] = singleowner['SystemCosts']['total_installed_cost'] / \
                                                       singleowner['FinancialParameters']['system_capacity']
            Wind['om_cost_us_dollars_per_kw'] = singleowner['SystemCosts']['om_capacity'][0]

            # pass in resource data so reopt doesn't have to redownload
            resource_file = self.tech_defaults['Wind']['Windpower']['Resource']['wind_resource_filename']
            if os.path.exists(resource_file):
                df_wind = pd.read_csv(resource_file, skiprows=5,
                                      names=['Temperature (C)', 'Pressure (atm)', 'Speed (m/s)', 'Direction (deg)'])

                # Truncate any leap day effects by shaving off December 31
                Wind['temperature_celsius'] = df_wind['Temperature (C)'].tolist()[0:8760]
                Wind['pressure_atmospheres'] = df_wind['Pressure (atm)'].tolist()[0:8760]
                Wind['wind_meters_per_sec'] = df_wind['Speed (m/s)'].tolist()[0:8760]
                Wind['wind_direction_degrees'] = df_wind['Direction (deg)'].tolist()[0:8760]

        return Wind

    @property
    def Storage(self):
        """ The Storage dictionary required by REopt"""

        return None

    @property
    def Generator(self):
        """ The Generator dictionary required by REopt"""

        Generator = dict()
        Generator['max_kw'] = 0
        return Generator

    @property
    def post(self):
        """ The HTTP POST required by REopt"""

        post = dict()
        post['Scenario'] = self.scenario
        post['Scenario']['Site'] = self.site
        post['Scenario']['Site']['ElectricTariff'] = self.tariff
        post['Scenario']['Site']['LoadProfile'] = self.load
        post['Scenario']['Site']['Generator'] = self.Generator
        post['Scenario']['Site']['Financial'] = self.financial

        if self.PV is not None:
            post['Scenario']['Site']['PV'] = self.PV
        else:
            post['Scenario']['Site']['PV'] = dict()
            post['Scenario']['Site']['PV']['max_kw'] = self.interconnection_limit_kw

        if self.Wind is not None:
            post['Scenario']['Site']['Wind'] = self.Wind
        else:
            post['Scenario']['Site']['Wind'] = dict()
            post['Scenario']['Site']['Wind']['federal_itc_ptc'] = 0.0
            post['Scenario']['Site']['Wind']['max_kw'] = self.interconnection_limit_kw
            post['Scenario']['Site']['Wind']['size_class'] = 'large'

        if self.Storage is not None:
            post['Scenario']['Site']['Storage'] = self.Storage


        # write file to results for debugging
        with open(os.path.join(self.path_results, 'post.json'), 'w') as outfile:
            json.dump(post, outfile)

        return json.dumps(post)


    @property
    def scenario(self):
        """ The scenario dictionary required by REopt"""
        scenario_dict = dict()
        scenario_dict['user_id'] = "hybrid_systems"
        return scenario_dict


    @property
    def site(self):
        """ The site dictionary required by REopt"""
        site_dict = dict()
        site_dict['latitude'] = self.latitude
        site_dict['longitude'] = self.longitude
        return site_dict


    @property
    def financial(self):
        """ The financial dictionary required by REopt"""
        singleowner_generic = self.tech_defaults['Generic']['Singleowner']

        financial_dict = dict()
        financial_dict['analysis_years'] = singleowner_generic['FinancialParameters']['analysis_period']
        financial_dict['offtaker_discount_pct'] = \
            (1 + singleowner_generic['FinancialParameters']['real_discount_rate']*0.01)* \
            (1 + singleowner_generic['FinancialParameters']['inflation_rate'] * 0.01) - 1
        financial_dict['offtaker_tax_pct'] = (singleowner_generic['FinancialParameters']['state_tax_rate'][0] + \
                                              singleowner_generic['FinancialParameters']['federal_tax_rate'][0])*0.01
        om_escal_nom = (1+singleowner_generic['SystemCosts']['om_capacity_escal']*0.01)* \
                       (1 + singleowner_generic['FinancialParameters']['inflation_rate'] * 0.01) - 1
        financial_dict['om_cost_escalation_pct'] = om_escal_nom
        financial_dict['escalation_pct'] = singleowner_generic['PPAPrice']['ppa_escalation'] * 0.01
        return financial_dict

    @property
    def urdb_response(self):
        """ Calls the URDB API to get a utility rate required by REopt"""

        ur = UtilityRate(path_rates=self.path_rates, urdb_label=self.urdb_label)
        return ur.get_urdb_response()


    @property
    def tariff(self):
        """The ElectricityTariff dict required by REopt."""

        tariff_dict = dict()

        if self.urdb_response is not None:
            tariff_dict['urdb_response'] = self.urdb_response
        else:
            tariff_dict['blended_annual_demand_charges_us_dollars_per_kw'] = 0
            tariff_dict['blended_annual_rates_us_dollars_per_kwh'] = self.wholesale_rate

        # currently assume a Single Owner Merchant plant not in a NEM scenario
        tariff_dict['net_metering_limit_kw'] = self.net_metering_limit_kw
        tariff_dict['wholesale_rate_us_dollars_per_kwh'] = self.wholesale_rate
        tariff_dict['wholesale_rate_above_site_load_us_dollars_per_kwh'] = self.wholesale_rate_above_site_load_us_dollars_per_kwh

        if self.interconnection_limit_kw is not None:
            tariff_dict['interconnection_limit_kw'] = self.interconnection_limit_kw

        return tariff_dict

    @property
    def load(self):
        """The ElectricLoad dict required by REopt"""

        load_dict = dict()
        if self.load_profile is None:
            self.load_profile = 8760 * [0.0]
            load_dict['loads_kw'] = self.load_profile
        return load_dict

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

        results = dict()
        success = os.path.isfile(self.fileout)
        if not success or force_download:
            post_url = self.reopt_api_post_url + '&api_key={api_key}'.format(api_key=self.api_key)
            resp = requests.post(post_url, self.post, verify=False)

            if resp.ok:
                run_id_dict = json.loads(resp.text)

                try:
                    run_id = run_id_dict['run_uuid']
                except KeyError:
                    msg = "Response from {} did not contain run_uuid.".format(post_url)
                    raise KeyError(msg)


                poll_url = self.reopt_api_poll_url + '{run_uuid}/results/?api_key={api_key}'.format(
                    run_uuid=run_id,
                    api_key=self.api_key)
                results = self.poller(url=poll_url)
                with open(self.fileout, 'w') as fp:
                    json.dump(obj=results, fp=fp)
            else:
                resp.raise_for_status()
        elif success:
            with open(self.fileout, 'r') as fp:
                results = json.load(fp=fp)

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