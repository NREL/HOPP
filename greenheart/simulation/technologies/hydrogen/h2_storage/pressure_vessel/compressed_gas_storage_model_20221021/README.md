# Compressed gas storage for wind-H2-steel project
- Author: Peng Peng (ppeng@lbl.gov)
- Date: 10/21/2022
- Brief description: This script is for a high-level overall estimation of energy consumption for compressed gas H2 storage for a steel facility requiring 200 tonnes of hydrogen per day. 

## Required files:
1.	Compressed_all.py
2.	Compressed_gas_function.py
3.	Tankinator_large.xlsx
## Key inputs:
1.	Wind availability in %
2.	Charge and discharge ratio
3.	Flow rate
4.	Energy (renewable) cost in $/kWh
## Key output:
1.	Maximum equivalent storage capacity.
2.	Maximum equivalent storage duration.
3.	Fitting parameters for further calculations and optimizations.
    1.	purchase capital storage cost ($/kg) vs. capacity
    2.	annual operation cost ($/kg) vs. capacity.  

![](images/2022-11-18-15-30-34.png)  
![](images/2022-11-18-15-31-18.png)
             
## How to use:
1.	Put all three files in one folder
2.	Change the path of the Tankinator file
3.	Change inputs and run Compressed_all.py

## Notes:
1.	Max storage capacity obtained from empirical relation from another project for 200 tonne/day H2 flow into steel facility. See below. And is assumed to scale linearly with the steel facility.
    - ![](images/2022-11-18-15-31-54.png)
2.	Operation cost does not include electrolyzer.
3.	Costs include the following dummy values, which can be changed later in the “Economic parameters” section (around line 64) in Compressed_gas_function.py 
    1.	Site preparation $100/kg H2 stored
    2.	80% markup for tank manufacturing based on materials
    3.	50% markup from purchase cost for engineering, installation, etc.
    4.	labor @ 36 $/h, 360 days x2 
    5.	annual maintenance @ 3% purchase capital 
4.	Capital cost is for 2021 CEPCI, which can be changed at the same place as above
5.	The code is for 350 bar compressed gas storage. If change pressure make sure to change the pressure in the Tankinator file and save it again before running.
6.	The main components included are storage tanks, compressors, refrigeration, and heater. 
7.	Calculations are based on high-level energy balances, detailed energy consumption should be performed with more comprehensive process simulation. 
8.	Heat transfer between the ambient, and friction losses are not included. 
9.	From the storage capacity (x) derived from the wind availability, the cost relation is derived from the Cost relation fitted from 0.1x to x. This is done because the fitting correlations do not work very well across a very large range (Let me know if you want to optimize for a larger range). This will lead to a small difference for the same capacity when the max changes. See the example below for different wind availabilities, the fitted values for 105 kg capacity are different. I tested different step sizes and ranges and found 0.1x-x, 15-25 steps tend to give the smallest difference for this fitting correlation.
![](images/2022-11-18-15-32-24.png)
![](images/2022-11-18-15-32-32.png)
10. Cost year for the model is 2021 with the chemical plant cost index set as 708. 

## Main references: 
1. Geankoplis, C. J. "Transport processes and separation." Process Principles. Prentice Hall NJ, 2003.
2. Dicks, A. L. & Rand, D. A. J. in Fuel Cell Systems Explained 351-399 (Wiley, 2018).
3. Turton, R., Bailie, R. C., Whiting, W. B., & Shaeiwitz, J. A. (2018). Analysis, synthesis and design of chemical processes. Pearson Education. 5th edition.
4. Luyben, W. L. (2017). Estimating refrigeration costs at cryogenic temperatures. Computers & Chemical Engineering, 103, 144-150.
