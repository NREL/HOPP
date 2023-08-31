## Intro

The functions in this directory model green steel production.  More specifically it models the mass/energy flows of a Hydrogen Direct Reduction Iron (HDRI) shaft and an Electric Arc Furnace (EAF) separately (The electrolyzer modeled in the example script but hopp results should replace the simple model). The results from the thermodynamic modeling are then used to make financial estimations on capital costs and operational costs.  The basis of these models are from Abhinav Bhaskar's paper “Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." doi: https://doi.org/10.1016/j.jclepro.2022.131339. 

Author: Charles Kiefer
Date:8/31/2023

## Overview of Plant

The main subsystems of the plant are the HDRI, EAF and Electrolyzer. At a high level, the main inputs of the plant are iron ore, carbon, water and electricity.

### HDRI

At a high level, the HDRI shaft converts iron ore (Fe2O3) into pure iron (Fe) using hydrogen gas (H2).  The hydrogen gas combines with the oxygen in the iron ore and produces water in the form of steam.

#### HDRI Detailed Level

At a more detailed level, hydrogen gas enters the shaft at 1173 K.  An electric heater and heat exchanger is used to achieve this temperature.  Enthalpy functions are utilized to determine the amount of energy needed to heat up hydrogen gas.

The main process of iron reduction in an HDRI-EAF system is the HDRI shaft.  The two main producers of DRI systems are MIDREX and Energiron (International Iron Metallics Association n.d.).  Each company has differences in their design, but the main reduction process is the same.  Iron ore enters a shaft furnace and is reduced by a reducing agent. Reducing agents are substances, usually gases, that bond with the oxygen in the ore.  Through this process, hematite (Fe2O3) reduces through many stages but finalizes itself as pure iron (Fe), also known as sponge iron.  In an HDRI system, the reducing agent is hydrogen. The chemical reactions of hydrogen reduction can be seen in equations (3), (4) and (5) (Patisson and Mirgaux 2020).

Fe_2 O_3 + 3H_2 = 2Fe + 3H_2 O                                          (3)

Fe_3 O_4 + 16/19 H_2 = 60/10 Fe_0.95 O + 16/19 H_2 O                    (4)

Fe_0.95 O + H_2 = .95Fe + H_2 O                                         (5)

Reduction rates with hydrogen are higher than that of carbon gas mixes.  In addition, higher hydrogen temperatures and pressures lead to increased reduction rates (El-Geassy and Rajakumar 1985). The model in this report assumes that the hydrogen was not pressurized; it remains at atmospheric pressure.  The temperature of the hydrogen is assumed to be at 1173 K.  Work done previously exhibits that these parameters are satisfactory but can be more efficient (Sato, Ueda and Yasunori 1986).

The model consists of two main inlets and two outlets.  One inlet is iron ore which consists of hematite and impurities. These impurities can consist of many elements.  Most of the impurities are other metallic oxides like CaO and SiO2.  This model assumes that the impurities of the ore only consist of SiO2 and Al2O3.

The other inlet in the model is the hydrogen. There is a stoichiometric amount required to reduce all the iron ore. However, a higher flow rate of hydrogen should be achieved as a buffer as some of the hydrogen will re-oxidize (Duarte and Pauluzzi 2019).  To accommodate for this, the actual flow rate of the model is 20% higher than the stoichiometric requirement.

The outlets in the system are the reduced iron ore and the exhaust of the hydrogen.  The amount reduced iron is dependent on the metallization rate of the HDRI shaft.  MIDREX shafts claim to have a metallization rate of 92%-96% (Midrex Technologies, Inc. 2018) (Sanjal 2015).  This model assumes a metallization rate of 94% with a temperature 973 K.  The exhaust consists of excess hydrogen and water, a product of reduction. The total mass of the water produced can be determined with the molecular weight ratios and mol ratios.  It is assumed that all the stoichiometrically required hydrogen is consumed and the resulting temperature is 573 K.

The following are additional assumptions of the HDRI system:

•   No heat loss in the reduced iron from the HDRI shaft to the EAF

•   Hydrogen apparent activation energy is 35 KJ/mol (Bhaskar, Mohsen and Homam, Decarbonization of the Iron and Steel Industry with Direct Reduction of Iron Ore with Green Hydrogen 2020)

•   100% hydrogen is fed into the HDRI shaft

•   Iron Pellets are not preheated

### EAF

The EAF acts as the primary steelmaker in the production process.  The concept of an EAF system is related to that of welding.  Large electrodes in the EAF run high-voltage open currents between them generating large amount of heat.  Most steel plants depend on the EAF to be the primary melting process.

#### EAF Detailed View

The reduced iron from the HDRI enters the EAF at a temperature of 973 K.  The model assumes that there is no heat loss in the transition of HDRI Shaft and the EAF.  Once in the furnace, the iron is heated to 1923 K and becomes molten.  The molten stream leaves the furnace at the same temperature.

Coke, lime, and oxygen are also added to the EAF to help reduce the FeO2, form a layer of slag, produce carbon-steel, and provide extra energy.  The coke performs additional reduction in the EAF, contributing about 4.4% more steel. However, the model does not take the excess into account financially and turns it into a buffer to accommodate of 4.4% steel loss.  Coke is also added to increase the carbon composition of the steel.

The added lime is usually some sort of chemical compound containing calcium or magnesium and oxygen. It acts as a slag former as the calcium bonds with the impurities in the iron stream.  This slag layer acts as an insulator, extracts unwanted impurities, and extends the electrode life (National Lime Association 2022). Slag leaves the system at a temperature of 1923 K.

For sake of simplicity, the added energy from inputted oxygen, commonly known as oxyfuel, was not considered.  The addition of oxyfuel would decrease the required electricity demand of the EAF and change the percent carbon in the steel.  With these additions, there is unfortunately some sort of emissions produced.  However, the emissions produced are extremely low compared to traditional routes.

The main reactions in the EAF are the following (Hornby and Brooks 2021):

FeO + C = Fe + CO                                                       (4)

FeO + CO = Fe + CO_2                                                    (5)

Fe + 1/2 O_2 = FeO                                                      (6)

These are the major reactions of most primary steel making processes.  The secondary metallurgy process is where the composition of the steel is formed. Depending on the desired composition of the steel, other materials and elements are added into the ladle furnace.  These can include the elements Si, Mn, P, S, Cr, Ni, Mo, and Cu (Camdali and Tunc 2016). These reactions look like the following (Yu, et al. 2021):

Mn + O = MnO                                                            (7)

Si + 2O = SiO_2                                                         (8)

This model did not take account in the secondary metallurgy process and subsequently these reactions.  It is important to note the energy needed to facilitate these reactions varies depending on the desired compositions.  Further work should be done to thermodynamically model the secondary metallurgy of each steel composition, but the scope of this work attains to a generalization of the EAF process.

As mentioned earlier, the reactions of the coke, lime and oxygen were not thermodynamically modelled. Instead, assumptions were made on minor reactions in the EAF. The following assumptions are based off the assumptions Abhinav Bhaskar made in his work to determine validation (Bhaskar, Abhishek, et al. 2022).

•   10 kg/tls of coke are injected

•   50 kg/tls of lime are injected

•   208 kg/tls of oxygen are injected

•   .113 tCO2/tls are generated

Other assumptions of the EAF include:

•   No scrap involved, 100% pellet fed

•   4.4% steel loss

•   EAF electrical efficiency is 60%

### Electrolyzer

Electrolyzer supplies the HDRI shaft with hydrogen and the EAF with oxygen.  The HDRI model determines how much hydrogen is needed and then uses simple conversions to determine the amount of electricity and water required.

## How the code works

Units are in the models.  The main input for the code is steel_output_desired which is in kg/hr.  The units commented throughout the model are in respect to kg/hr.  It is possible to use units like tls/hr or tls, but it is not advised as the units may need to be tracked through to calculate locos.

### Example run script

The overview on how to use the code is found in example_steel_run_script.py.

The first function is greensteel_run(): which identifies how to run the models and pull the individual data from each model.  It is recommended that each class is given an instance variable.  Then the model needed should be pulled from this variable.
For example:

```
eaf_model_instance = eaf_model() 

eaf_mass_outputs = eaf_model_instance.mass_model(steel_output_desired_kg_hr)

eaf_energy_outputs = eaf_model_instance.energy_model(steel_output_desired_kg_hr)

eaf_emission_outputs = eaf_model_instance.emission_model(steel_output_desired_kg_hr)

eaf_financial_outputs = eaf_model_instance.financial_model(steel_out_year_tls)
```

The second function, h2_main_steel():, is how to use the models to calculate total costs and a method to calculate lcos.  

### Individual models

Each plant is defined by a class.  The classes are then broken into smaller models for the plant.  The smaller models are a mass, energy and financial for both as well as an __init__.  The HDRI class also holds the heater and recuperator (heat exchanger) model.  The EAF holds an emission model.

The models of HDRI and EAF have __init__ functions that hold all the assumptions needed for the plant.  For example, all the needed temperatures of each stream are stored in the __init__.  Financial assumptions are held here for each respective subsystem.  Also, __init__ shows what is being calculated and the units for each calculation.

The most common input of the models is steel_out_desired in kg/hr and steel_prod_yr in tls/yr. 

As well enthalpy_functions.py holds all the enthalpy formulas used to calculate the energy equation.

### hdri_model.py

The main model that drives is the mass_model.  The mass_model takes steel_out_desired and determines how much hydrogen and iron ore is needed.  Then it calculates the compositions of the outlets. There are 2 inlets and 2 outlets.  The inlets are pure H2 gas and iron ore.  The outlets are a H2/H2O stream and reduced iron.

The energy_model runs the mass_model to gather the outlets and calculates the energy of the HDRI system using the enthalpy functions.  This should always return a negative value.  The negative value denotes heat is leaving the system.  Mostly used as a check on the mass_model.

The financial_model runs the mass_model with steel_prod_yr as the argument to determine yearly masses for materials.  Capital cost is modeled from the yearly capacity of the plant and subsequently other costs are dependent on the capital cost.  The costs calculated include capital, operational, maintenance, depreciation, iron ore, and labor.

The recuperator_model runs the mass_model and energy_model to calculate masses and enthalpies to determine the enthalpy entering the heater. 

The last model is the heater_mass_energy_model.  This model runs the mass_model and energy_model as well and calculates the needed energy to raise H2 gas to the set temp of the inlet stream.

### eaf_model.py

Very similar to hdri_model.py

The driving model is the mass_model.  the model takes in steel_out_desired and calculates all the masses of the primary inlets and outlets.  It does not calculate the slag produced or the oxygen inputted into the EAF.  The main inlet is the steel produced and the main inlets are iron, carbon and lime.  The model calculates the actual steel output.  There is about a 4.4% increase in steel produced as there is unintentional iron ore oxidation that in turn produces a higher mass.

The energy_model uses the mass_model calculations to determine the energy needed.  This model uses enthalpy_functions and uses an efficiency factor stored in the __init__.  It returns kwh for steel output per hour as the units.  The example script shows the conversion to get a value in kwh/year.

The emission model pulls in the energy_model and the heater_mass_energy_model total energy values to calculate total direct emissions and indirect emissions.  The indirect emissions are emissions from the energy pulled from the grid and are not usually paid for by the plant.

The financial model does not need to pull in other models as the costs are all associated with the capacity of the plant which is the argument.  The model calculates capital, operational, maintenance, depreciation, coal, lime, emission and labor costs.

## Sources

Bhaskar, Abhinav, Assadi Mohsen, and Somehsaraei Nikpey Homam. 2020. "Decarbonization of the Iron and Steel Industry with Direct Reduction of Iron Ore with Green Hydrogen." Energies 13 (3): 758. doi: https://doi.org/10.3390/en13030758 

Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339. 

Camdali, U., and M. Tunc. 2016. "Calculation of Chemical Reaction Energy in an Electric Arc Furnace and Ladle Furnace System." Metallurgist (9): 669-675. doi: https://doi.org/10.1007/s11015-016-0349-9 

Duarte, Pablo, and Dario Pauluzzi. 2019. "Premium Quality DRI Products." https://www.energiron.com/wp-content/uploads/2019/07/Premium-Quality-DRI-Products-from-ENERGIRON.pdf 

El-Geassy, A.A., and V Rajakumar. 1985. "Gaseous Reduction of Wustite with H2, CO and CO Mixtures*." Trans. Iron Steel Institute Japan 25: 449-458. doi: https://doi.org/10.2355/isijinternational1966.25.449 

Hornby, Sara, and Geoff Brooks. 2021. "Impact of Hydrogen DRI on EAF Steelmaking." Midrex: Iron and Steel Industry. https://www.midrex.com/tech-article/impact-of-hydrogen-dri-on-eaf-steelmaking/ 

International Iron Metallics Association. n.d. DRI Production. IIMA. Accessed July 26, 2022. https://www.metallics.org/dri-production.html. 

Midrex Technologies, Inc. 2018. "DRI Products + Applications: Providing flexibility for steelmaking." April. Accessed May 2, 2022. https://www.midrex.com/wp-content/uploads/MIdrexDRI_ProductsBrochure_4-12-18.pdf.

National Lime Association. 2022. Lime Basics. National Lime Association. Accessed July 25, 2022. https://www.lime.org/lime-basics/uses-of-lime/metallurgical-uses-of-lime/iron-and-steel/.

Ranzani da Costa, A, D Wagner, and F Patisson. 2013. "Modelling a new, low CO2 emissions, hydrogen steelmaking process." Journal of Cleaner Production 46: 27-35. doi: https://doi.org/10.48550/arXiv.1402.1715  

Sanjal, Sujit. 2015. "The Value of DRI – Using the Product for Optimum Steelmaking." MIDREX. https://www.midrex.com/tech-article/the-value-of-dri-using-the-product-for-optimum-steelmaking/ 

Sato, Kyoji, Yoshinobu Ueda, and Nishikawa Yasunori. 1986. "Effect of Pressure on Reduction Rate of Iron Ore with High Pressure Fluidized Bed*." Trans. Iron Steel Institute Japan 26: 697-703. doi: https://doi.org/10.2355/isijinternational1966.26.697 