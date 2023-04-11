# from hybrid.PEM_H2_LT_electrolyzer import PEM_electrolyzer_LT
# from numpy.lib.function_base import average
# import examples.H2_Analysis.H2AModel as H2AModel
import numpy as np
import pandas as pd
from power_electronic_info import *
#from HOPP.tools.resource import resource
#from hOPP.hybrid.add_custom_modules.custom_wind_floris import custom_wind_floris
#from hybrid.add_custom_modules.custom_wind_floris import Floris
#from hybrid.pv_source import PVPlant
#states=['IN','IA','MS','TX','WY']

# pe_folder = '/Users/egrant/Desktop/HOPP-GIT/PowerElecResults/Central/'
# pe_folder = '/Users/egrant/Desktop/HOPP-GIT/PowerElecResults/Distributed/'
# loss=[]
# loss_oi=['Rectifier Loss [%]','Transformer Loss [%]','Cable Loss [%]']
# loss_var=loss_oi[0]
# loss_b4='Inital Power [MW] with Default Losses'
# loss_after='Power After Rectifier [MW]'
# power_before=[]
# power_after=[]
# for state in states:
#       # df=pd.read_csv(pe_folder + state + '_loss_data_central.csv',index_col='Unnamed: 0')
#       df=pd.read_csv(pe_folder + state + '_power_data_distributed.csv',index_col='Unnamed: 0')
#       power_before=df.loc[loss_b4]
#       power_after=df.loc[loss_after]
#       ploss = (power_after-power_before)/power_after
#       loss.append(ploss.values)
#       # loss.append(df.loc[loss_var].values)
# []
def turbine_losses_floris(farm_power_kW,nTurbs,turbine_rating_kW,electrolyzer_size_kW,electrolysis_scale):
      
      avg_cable_loss_central=2.578824156495309 #[%]
      #below is for comparison to previous test-cases
      #when wind farm power was re-calculated
      avg_transformer_loss_central=3.0673368741057407 #[%]
      avg_rectifier_loss_central=6.650622292138879
      avg_rectifier_loss_distributed=5.56452488812321

      pysam_elec_losses=1.91
      farm_rating_kW = nTurbs*turbine_rating_kW
      load_rating_kW=electrolyzer_size_kW
      rectifier_size = np.max([load_rating_kW,farm_rating_kW])
      transformer_size = 1.2*farm_rating_kW

      #central
      power_generated = farm_power_kW*((100+pysam_elec_losses)/100)
      transformer=ac2ac_transformer(transformer_size)
      rectifier=ac2dc_unidirect_rectifier(rectifier_size)
      #ac2ac transformer losses
      if electrolysis_scale=='Centralized':
            power_t1=power_generated*transformer.f(power_generated/transformer_size)
            #average cable losses
            power_c1=power_t1*((100-avg_cable_loss_central)/100) 
            ##ac2dc rectifier losses
            power_r1=power_c1*rectifier.f(power_c1/rectifier_size) 
      else:
            #rectifier loss
            #dc power output
            power_r1= power_generated*rectifier.f(power_generated/rectifier_size) 
      return power_r1 #turbine power in kWdc for PEM

def calc_power_loss(initial_power,final_power):
      power_diff=np.sum(final_power) - np.sum(initial_power)
      power_percent_loss = power_diff/np.sum(initial_power)
      print("{}% Power loss from power electronic components".format(round(-1*power_percent_loss,3)))
