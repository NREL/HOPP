from hybrid.sites import SiteInfo
import os
import json
from dotenv import load_dotenv
from hybrid.keys import set_developer_nrel_gov_key
from hybrid.sites import flatirons_site as sample_site
import post_and_poll
import results_poller
import reopt_logger
import pandas as pd

load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# Load the sample JSON
f = open('massproducer_offgrid (1).json')
data_for_post = json.load(f)

# Create a post
post_and_poll.get_api_results(data_for_post, NREL_API_KEY, 'https://offgrid-electrolyzer-reopt-dev-api.its.nrel.gov/v1',
                              'reopt_result_test.json')

# Poll for results
# results_poller.poller('https://offgrid-electrolyzer-reopt-dev-api.its.nrel.gov/v1', 5)