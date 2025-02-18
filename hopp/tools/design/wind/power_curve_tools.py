import numpy as np
import matplotlib.pyplot as plt
import copy

def plot_power_curve(wind_speeds_ms,cp_curve,ct_curve):
    fig1 = plt.figure()
    plt.plot(wind_speeds_ms, ct_curve, label="Coeff of Thrust")
    plt.plot(wind_speeds_ms, cp_curve, label = "Coeff of Power")
    plt.legend()
    plt.xlabel("Wind Speed [m/s]")
    plt.show()


def pad_power_curve(wind_speed,curve,v_min = 0.0,v_max = 50.0):
    wind_speeds_ms = copy.deepcopy(wind_speed)
    if isinstance(wind_speeds_ms,list):
        wind_speeds_ms = np.array(wind_speeds_ms)
    if isinstance(curve,list):
        curve = np.array(curve)
    if min(wind_speeds_ms) > v_min:
        wind_speed_pad = np.arange(v_min,min(wind_speeds_ms),1)
        wind_speeds_ms = np.concatenate((wind_speed_pad,wind_speeds_ms))
        curve = np.concatenate((np.zeros(len(wind_speed_pad)),curve))

    if max(wind_speeds_ms) < v_max:
        wind_speed_pad = np.arange(max(wind_speeds_ms),v_max,1)
        wind_speeds_ms = np.concatenate((wind_speeds_ms,wind_speed_pad))
        curve = np.concatenate((curve,np.zeros(len(wind_speed_pad))))
    return list(wind_speeds_ms), list(curve)

def calculate_cp_from_power(wind_speeds_ms,power_curve_kw,rotor_diameter,air_density = 1.225):
    rotor_area = np.pi*((rotor_diameter/2)**2)
    if isinstance(wind_speeds_ms,list):
        wind_speeds_ms = np.array(wind_speeds_ms)
    if isinstance(power_curve_kw,list):
        power_curve_kw = np.array(power_curve_kw)
    # power available in the wind (kW)
    p_wind = 0.5*air_density*rotor_area*(wind_speeds_ms**3)/1e3
    cp = power_curve_kw/p_wind
    cp = np.where(cp<0,0,cp)
    return list(cp)

def calculate_power_from_cp(wind_speeds_ms,cp_curve,rotor_diameter,rated_power_kW, air_density = 1.225):
    rotor_area = np.pi*((rotor_diameter/2)**2)
    if isinstance(wind_speeds_ms,list):
        wind_speeds_ms = np.array(wind_speeds_ms)
    if isinstance(cp_curve,list):
        cp_curve = np.array(cp_curve)
    # power available in the wind (kW)
    p_wind = 0.5*air_density*rotor_area*(wind_speeds_ms**3)/1e3
    power_kW = cp_curve*p_wind
    power_kW = np.where(power_kW>rated_power_kW,rated_power_kW,power_kW)
    power_kW = np.where(power_kW<0,0,power_kW)
    return list(power_kW)

def estimate_thrust_coefficient(wind_speeds_ms,cp_curve, plot=False, print_output=False):

    # Check that the wind speed and the coefficient of power are the same length
    if len(wind_speeds_ms) != len(cp_curve):
        print("The length of the wind speed and coefficient of power vectors must be the same")
        return

    # wind_speeds_ms = wind_speeds_ms.dropna()
    # cp_curve = cp_curve.dropna()
        
    N_wind = len(wind_speeds_ms)
    ct_curve = list(np.zeros(N_wind))
    
    for i in range(N_wind):
        # calculate induction factor a
        # solve C_P = 4 * a * (1-a)**a  -> 4 * a**3 - 8 * a**2 + 4 * a - C_P = 0
        p = np.zeros(4)
        p[0] = 4
        p[1] = -8
        p[2] = 4
        p[3] = -cp_curve[i]
        roots = np.roots(p)

        # Take root that is in range of a -> [0, 0.5]
        a = roots[np.where(np.logical_and(roots>= 0, roots<= 0.5))]

        # Calculate C_T = 4 * a * (1-a)
        ct_curve[i] = np.round(4 * a * (1-a), 4)
        # print(coefficient_of_thrust[i], coefficient_of_power[i]/(1-a))    # Equivalent calculation
  

    if plot:
        plot_power_curve(wind_speeds_ms,cp_curve,ct_curve)

    if print_output:
        print("Coefficient of Thrust: ")
        for i in ct_curve:
            print("-",i)
        print("Coefficient of Power: ")
        for i in cp_curve:
            print("-",i)
        print("Wind Speed: ")
        for i in wind_speeds_ms:
            print("-",i)
    ct_flat = [ct[0] for ct in ct_curve]
    return ct_flat