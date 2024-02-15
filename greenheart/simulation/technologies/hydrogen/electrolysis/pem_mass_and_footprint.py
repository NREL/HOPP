import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def _electrolyzer_footprint_data():
    """
    References:
    [1] Bolhui, 2017 https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2020/Dec/IRENA_Green_hydrogen_cost_2020.pdf
        - appears to include BOS
    [2] Bourne, 2017
        - 
    [3] McPHy, 2018 (https://mcphy.com/en/equipment-services/electrolyzers/)
    [4] Air Liquide 2021, Becancour Quebec
    """

    rating_mw = np.array([300, 100, 100, 20]) # [1], [2], [3], [4]
    footprint_sqft = np.array([161500, 37700, 48500, 465000]) # [1], [2], [3], [4]
    sqft_to_m2 = 0.092903
    footprint_m2 = footprint_sqft*sqft_to_m2

    return rating_mw, footprint_m2

def footprint(rating_mw):

    """
    Estimate the area required for the electrolyzer equipment using a linear scaling
    """
    
    footprint_m2 = rating_mw*48000*(1/1E3) # from Singlitico 2021, Table 1 (ratio is given in m2/GW, so a conversion is used here for MW)
    
    return footprint_m2

def _electrolyzer_mass_data():
    """
    References:
    [1] https://www.h-tec.com/en/products/detail/h-tec-pem-electrolyser-me450/me450/
    [2] https://www.nrel.gov/docs/fy19osti/70380.pdf
    """

    rating_mw = np.array([1, 1.25, 0.25, 45E-3, 40E-3, 28E-3, 14E-3, 14.4E-3, 7.2E-7])
    mass_kg = np.array([36E3, 17E3, 260, 900, 908, 858, 682, 275, 250])

    return rating_mw, mass_kg

def _electrolyzer_mass_fit(x, m, b):

    y = m*x + b

    return y

def mass(rating_mw):
    """
    Estimate the electorlyzer mass given the electrolyzer rating based on data.

    Note: the largest electrolyzer data available was for 1.25 MW. Also, given the current fit, the mass goes negative for very small electrolysis systems
    """

    rating_mw_fit, mass_kg_fit = _electrolyzer_mass_data()

    (m, b), pcov = curve_fit(_electrolyzer_mass_fit, rating_mw_fit, mass_kg_fit)

    mass_kg = _electrolyzer_mass_fit(rating_mw, m, b)

    return mass_kg

if __name__ == "__main__":

    fig,ax = plt.subplots(1,2)
    rating_mw, footprint_m2 = _electrolyzer_footprint_data()
    ax[0].scatter(rating_mw, footprint_m2, label="Data points")

    ratings = np.arange(0,1000)
    footprints = footprint(ratings)
    
    ax[0].plot(ratings, footprints, label="Scaling Factor")
    ax[0].set(xlabel="Electrolyzer Rating (MW)", ylabel="Footprint (m$^2$)")
    ax[0].legend(frameon=False)
    print(rating_mw, footprint_m2)

    rating_mw, mass_kg = _electrolyzer_mass_data()
    ax[1].scatter(rating_mw, np.multiply(mass_kg, 1E-3), label="Data points")

    ax[1].plot(ratings, mass(ratings)*1E-3, label="Linear Fit")
    ax[1].set(xlabel="Electrolyzer Rating (MW)", ylabel="Mass (tonnes)")
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.show()
    print(rating_mw, np.divide(mass_kg, rating_mw))