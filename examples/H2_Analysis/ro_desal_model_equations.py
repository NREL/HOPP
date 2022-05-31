
from urllib.parse import quote_from_bytes


membrane_flow_rate = 100  # m^3/s
membrane_specific_permeability = 99   # m^2/h-Pa
surface_of_membrane = 0  # m^2
water_temp = 0

osmotic_pressure = (((0.6955 + 0.0025*water_temp)*10**(-8)) / seawater_density)*(membrane_salt_conc - premeate_salt_conc)

#  = membrane_flow_rate / surface_of_membrane - feed_channel_pressure_drop)

# Permeate Flux
permeate_flux = membrane_specific_permeability*(delta_pressure_feedwater_permeate - osmotic_pressure)

# Solvent permeability coefficient
membrane_specific_permeability = (water_diffusivity * membrane_water_conc * molar_volume_water) / (membrane_thickness * gas_constant * temp)

(D_w*C_w*V_w) / (del_m * R *T)


# Solute transport
J_s = B_s*(C_m - C_p)
# Solute permeability coefficient
B_s = D_s*K_s / del_m

# Salt rejection
R_s = (1 + (B_s/(A_w *(delta_rho - osmotic_pressure)))**(-1))
# Osmotic pressure
osmotic_pressure = R*T*sum(n/v)

# Temperature correction factor
TCF = exp((E_m/R)*((1/273 + T) - (1/298)))

# Specific energy
E = (P_f*Q_f*(E_pump)**(-1) - (P_r*Q_r*E_ERD))/Q_p

# Recovery ratio
R = Q_p / Q_f
# Total mass balance
# Q_f*C_f = Q_p*C_p + Q_r*C_r

# Pressure applied across membrane
delta_P = (P_f + P_r)/2 - P_p

""" 
Create an instance of a high-pressure reverse osmosis desalination
system. 

Parameters
_____________
Add parameters here

Returns
_____________
Add returns here
Model is used to determine flow rate profile and associated Capex and Opex of High Pressure Osmosis Desalination

inputs include: base case - flow rate and operational parameters
Outputs include: flow rate profile, Capex, Opex

expanded model: cold/warm start, need for continuous power, additional operation parameters
""" 