# This file is part of CuLPy
# Copyright (c) 2024 Burak Kaynaroglu
# This program is free software distributed under the MIT License 
# A copy of the MIT License can be found at 
# https://github.com/kaynarob/CuLPy/blob/main/LICENSE.md

""" CuLPY model for 0-dimentional configuration """


import math
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


start_time = time.time()
# =========================================================================== #
# Case specific properties, file names, initialization values /
# =========================================================================== #
H_CL  =                  # Average water depth
Altitude =               # Site specific altitude (m).
sim_start_date = ""      # Simulation start date
sim_end_date = ""        # Simulation end date
JDay_start_date = ""     # Input starting Julian day 
dt =                     # time step in days
kmc_file_name = ""       # model parameter file name and path
# initial concentrations 
state_vars_init_dict1 = {"Cpy": ,  
                        "Cpoc": ,  "Cpon": , "Cpop": ,
                        "Cdoc": ,  "Cdon": , "Cdop": ,
                         "Cam": ,   "Cni": ,  "Cph": , 
                         "Cox": }

# =========================================================================== #
# Case specific properties \
# =========================================================================== #

print('\n# =================================================================== #')
print('\nmodel initialization...\n')
total_boxNo = 1
print(f'\tnumber of boxes                : {total_boxNo}')
sediment_option = 0
print(f'\tsediment_option                : {int(sediment_option)}')
print(f'\tpelagic constants file name    : {kmc_file_name}')
print(f'\tsediment constants file name   : {smc_file_name}')
print(f'\tsimulation start date          : {sim_start_date}')
print(f'\tsimulation end date            : {sim_end_date}')
print(f'\ttime step in days              : {int(1/dt)}')
print(f'\tdt                             : {int(24*60*dt)} minute(s)')

# =========================== number of iteration =========================== #
date_1 = datetime.strptime(sim_start_date, '%Y-%m-%d')
date_2 = datetime.strptime(sim_end_date, '%Y-%m-%d')
sim_end_jdays = (date_2 - date_1).days
n_iter = int(sim_end_jdays / dt)  
# =========================== number of iteration =========================== #

def kmc_reader(file_name):
    kmc_dict = {}
    f = open(file_name, "r+")
    for line in f:
        line_ = line.split()
        kmc_name = line_[0]
        kmc_values = float(line_[2])
        kmc_dict[kmc_name] = kmc_values
    f.close()
    return kmc_dict

def interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, time_step_per_day, file_name):
    dt_minutes = 24 * 60 / time_step_per_day
    inputF_path = f'{file_name}.csv'
    inputF = pd.read_csv(inputF_path)
    start_date = datetime.strptime(JDay_start_date, "%Y-%m-%d")
    inputF['datetime'] = inputF['time'].apply(lambda x: start_date + timedelta(days=x))
    inputF.set_index('datetime', inplace=True)
    # Create new time index for interpolation
    new_time_index = pd.date_range(start=sim_start_date, end=sim_end_date, freq=f'{int(dt_minutes)}min')
    # Interpolate bc_flow data to new time index
    interpolated_input = inputF.reindex(inputF.index.union(new_time_index)).interpolate(method='time').loc[new_time_index]
    return interpolated_input

def interpolate_wDate(sim_start_date, sim_end_date, time_step_per_day, file_name):
    dt_minutes = 24 * 60 / time_step_per_day
    inputF_path = f'{file_name}.csv'
    inputF = pd.read_csv(inputF_path)
    inputF['time'] = pd.to_datetime(inputF['time'])
    inputF.set_index('time', inplace=True)
    new_time_index = pd.date_range(start=sim_start_date, end=sim_end_date, freq=f'{int(dt_minutes)}min')
    # Interpolate bc_concentration data to new time index
    if isinstance(inputF.index, pd.DatetimeIndex):
        interpolated_input = inputF.reindex(inputF.index.union(new_time_index)).interpolate(method='time').loc[new_time_index]
    else:
        raise ValueError("input does not have a DatetimeIndex, cannot perform time-based interpolation.")
    return interpolated_input

# =========================================================================== #
# ========================= Read/Interpolate Data \ ========================= #

# ============================ Model parameter \ ============================ #
kmc = kmc_reader(kmc_file_name)  # pelagic compartment parameters
# ============================ Model parameter / ============================ #

# Read and interpolate data 
# csv file name and path without extention must given
df_input_C01_Ri = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, "")  
df_input_C01_BS = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, "")
df_input_Q = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "")
df_input_T = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "")
df_input_V = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "")
df_input_Ia   = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "")
df_input_fDay = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "")
df_input_Salt = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "")

# Box1 CL Input Arrays
Q01_Ri    = df_input_Q['']*86400  # related column name in csv file must given
Q01_BS    = df_input_Q['']*86400  # related column name in csv file must given
Q10_BS    = df_input_Q['']*86400  # related column name in csv file must given
C01_Ri    = {var: df_input_C01_Ri[var] for var in state_vars_init_dict1}
C01_BS    = {var: df_input_C01_BS[var] for var in state_vars_init_dict1}
T         = df_input_T['']        # related column name in csv file must given
V         = df_input_V['']        # related column name in csv file must given
I_a       = df_input_Ia['']       # related column name in csv file must given
f_day     = df_input_fDay['']     # related column name in csv file must given
salinity  = df_input_Salt['']     # related column name in csv file must given

# ========================= Read/Interpolate Data / ========================= #
# =========================================================================== #

# =========================================================================== #
# ================== Numerical Solution for C dictionary \ ================== #

def simulate_C():

    C1 = {var: np.zeros(n_iter + 1) for var in state_vars_init_dict1}
    for var in state_vars_init_dict1:
        C1[var][0] = state_vars_init_dict1[var]
    
    for t in range(1, n_iter + 1):
        
        if sediment_option == 0:
            
            C1t  = {key: value[t-1] for key, value in C1.items()}
            R1_t = pelagic_process_rates(C1t, T[t-1], V[t-1], H_CL, I_a[t-1], salinity[t-1], f_day[t-1])
            
            for var in state_vars_init_dict1:
                
                C1[var][t] =  + C1[var][t-1] + (
                              + (Q01_Ri[t-1] / V[t-1]) * C01_Ri[var][t-1]
                              + (Q01_BS[t-1] / V[t-1]) * C01_BS[var][t-1]
                              - (Q10_BS[t-1] / V[t-1]) * C1[var][t-1]
                              + R1_t[var]
                              ) * dt
               
    return C1

# ================== Numerical Solution for C dictionary / ================== #
# =========================================================================== #

# =========================================================================== #
# =================== Pelagic process rates calculation \ =================== #

def pelagic_process_rates(C_t_dict, T, V, H, I_a, salinity, f_day):
    
    Cpy, Cpoc, Cpon, Cpop, Cdoc, Cdon, Cdop, Cam, Cni, Cph, Cox = list(C_t_dict.values())
    
    
    def calculate_light_limitation(ChlA, H, kmc, I_a):
        K_e = kmc['K_be'] + (0.0088*ChlA) + (0.054*(ChlA**(2/3)))
        constant_e = 2.718
        X_I = (((constant_e*f_day)/(K_e*H)) * 
                (math.exp((-I_a/kmc['I_s'])*math.exp(-K_e*H)) - 
                 math.exp(-I_a/kmc['I_s'])))
        return X_I
    
    def calculate_nutrient_limitation(Cni, Cam, Cph, kmc):
        X_N_N = (Cni+Cam)/(kmc['K_SN']+Cni+Cam)
        X_N_P = Cph/(kmc['K_SP']+Cph)
        return X_N_N*X_N_P

    def calculate_O2_saturation(T, kmc, salinity):
        TKelvin = T + 273.15
        AltEffect = (100 - (0.0035 * 3.28083 * Altitude)) / 100
        ln_stemp = (-139.34411 + (1.575701E+5 / TKelvin) - 
                    (6.642308E+7 / (TKelvin ** 2)) + (1.243800E+10 / (TKelvin ** 3)) 
                    - (8.621949E+11 / (TKelvin ** 4)))
        ln_ssalt = (salinity * ((1.7674E-2) - (1.754E+1 / TKelvin) 
                            + (2.1407E+3 / (TKelvin ** 2))))
        O2_sat_fresh = math.exp(ln_stemp)
        O2_sat_salt = math.exp(ln_ssalt)
        
        return AltEffect * (O2_sat_fresh - O2_sat_salt)
    
    
    ChlA = (Cpy / kmc['a_C_chl']) * 1000  # biomass as chlorophyll-a (µg/L) 
    X_I = calculate_light_limitation(ChlA, H, kmc, I_a)
    X_N = calculate_nutrient_limitation(Cni, Cam, Cph, kmc)
    # Cpy: Phytoplankton-Carbon processes
    r_Phyto_Death_Mortality = kmc['k_mortality'] * kmc['theta_mortality']**(T-20) * Cpy
    r_Phyto_Death_Salinity = kmc['k_salt_death'] * (salinity/(salinity+kmc['K_Sl_salt'])) * Cpy
    R_Cpy_Growth = (kmc['k_growth'] * (kmc['theta_growth']**(T - 20)) 
                     *(Cox/(Cox+kmc['K_Sl_ox_Cpy'])) * X_I * X_N * Cpy)
    R_Cpy_Respiration = kmc['k_resipration'] * kmc['theta_resipration']**(T-20) * Cpy
    R_Cpy_Excration = kmc['k_excration'] * kmc['theta_excration']**(T-20) * Cpy
    R_Cpy_Death = (r_Phyto_Death_Mortality + r_Phyto_Death_Salinity)
    R_Cpy_Settling = (kmc['v_set_Cpy']/H) * Cpy
    R_Cpy = (+R_Cpy_Growth
             -R_Cpy_Respiration
             -R_Cpy_Excration
             -R_Cpy_Death
             -R_Cpy_Settling)

    # Cpoc: Particulate organic carbon processes
    R_Cpoc_Decomposition = kmc['k_c_decomp'] * (kmc['theta_c_decomp']**(T-20)) * (Cpoc/(Cpoc+kmc['K_Sl_Cpoc_decomp'])) * Cpoc
    R_Cpoc_Settling = (kmc['v_set_Cpoc']/H) * Cpoc
    R_Cpoc = (+R_Cpy_Death
              -R_Cpoc_Decomposition
              -R_Cpoc_Settling)
   
    # Cpon: Particulate organic nitrogen processes
    R_Cpon_Decomposition = kmc['k_n_decomp'] * (kmc['theta_n_decomp']**(T-20)) * (Cpon/(Cpon+kmc['K_Sl_Cpon_decomp'])) * Cpon
    R_Cpon_Settling = (kmc['v_set_Cpon']/H) * Cpon
    R_Cpon = (+kmc['a_N_C']*R_Cpy_Death
             -R_Cpon_Decomposition
             -R_Cpon_Settling)
    
    # Cpop: Particulate organic phosphorous processes
    R_Cpop_Decomposition = kmc['k_p_decomp'] * (kmc['theta_p_decomp']**(T-20)) * (Cpop/(Cpop+kmc['K_Sl_Cpop_decomp'])) * Cpop
    R_Cpop_Settling = (kmc['v_set_Cpop']/H) * Cpop
    R_Cpop = (+kmc['a_P_C']*R_Cpy_Death
             -R_Cpop_Decomposition
             -R_Cpop_Settling)
    
    # Mineralization of dissolved organic maters by using oxygen 
    def mineralization_by_ox(var_name, var):
        return kmc[f'k_{var_name}_mnr_ox'] * (kmc[f'theta_{var_name}_mnr_ox']**(T-20)) * \
            Cox/(kmc[f'K_Sl_ox_mnr_{var_name}']+Cox) *\
                (var/(var+kmc[f'K_Sl_{var_name}_mnr_ox'])) * var
    
    # Mineralization of dissolved organic maters by using nitrate
    def mineralization_by_ni(var_name, var):
        return kmc[f'k_{var_name}_mnr_ni'] * (kmc[f'theta_{var_name}_mnr_ni']**(T-20)) * \
            (1 - Cox/(kmc[f'K_Si_ox_mnr_{var_name}']+Cox)) * \
                Cni/(kmc[f'K_Sl_ni_mnr_{var_name}']+Cni) * \
                    (var/(var+kmc[f'K_Sl_{var_name}_mnr_ni'])) * var
    
    # Cdoc: Dissolved organic carbon processes
    R_Cdoc_Mineralization = mineralization_by_ox("c", Cdoc) + mineralization_by_ni("c", Cdoc)
    R_Cdoc = (+R_Cpy_Excration
             +R_Cpoc_Decomposition
             -R_Cdoc_Mineralization)
   
    # Cdon: Dissolved organic nitrogen processes
    R_Cdon_Mineralization = (mineralization_by_ox("n", Cdon) + mineralization_by_ni("n", Cdon))
    R_Cdon = (+kmc['a_N_C']*R_Cpy_Excration
             +R_Cpon_Decomposition
             -R_Cdon_Mineralization)
    
    # Cdop: Dissolved organic phosphorous processes
    R_Cdop_Mineralization = (mineralization_by_ox("p", Cdop) + mineralization_by_ni("p", Cdop))
    R_Cdop = (+kmc['a_P_C']*R_Cpy_Excration
             +R_Cpop_Decomposition
             -R_Cdop_Mineralization)
    
    # Cam: Ammonia processes
    prefam = (Cam * (Cni / ((kmc['K_SN']+Cam)*(kmc['K_SN']+Cni))) + Cam * (kmc['K_SN'] / ((Cam+Cni)*(kmc['K_SN']+Cni))))
    R_Nitrification = (kmc['k_nitrification'] * (kmc['theta_nitr']**(T-20)) * (Cox/(Cox+kmc['K_Sl_nitr_ox'])) * (Cam/(Cam+kmc['K_Sl_nitr']))) * Cam
    R_Cam = (+R_Cdon_Mineralization
             +kmc['a_N_C']*R_Cpy_Respiration
             -kmc['a_N_C']*prefam*R_Cpy_Growth
             -R_Nitrification)
    
    # Cni: Nitrate processes
    R_Denitrification = ((kmc['k_denitrification'] * (kmc['theta_denitr']**(T-20)) * 
                          (kmc['K_Si_denitr_ox']/(Cox+kmc['K_Si_denitr_ox'])) * (Cni/(Cni+kmc['K_Sl_denitr']))) * Cni)
    R_Cni = (+R_Nitrification
             -R_Denitrification 
             -kmc['a_N_C']*(1-prefam)*R_Cpy_Growth)
    
    # Cph: Phosphate processes
    R_Cph = (+R_Cdop_Mineralization
             +kmc['a_P_C']*R_Cpy_Respiration
             -kmc['a_P_C']*R_Cpy_Growth)
    
    # Cox: Dissolved oxygen processes
    O2_sat = calculate_O2_saturation(T, kmc, salinity)
    R_Reaeration = kmc['k_raer'] * kmc['theta_rear'] * (O2_sat-Cox)
    R_Cox = (+R_Reaeration
            +kmc['a_O2_C']*R_Cpy_Growth
            -kmc['a_O2_C']*R_Cpy_Respiration 
            -(32/12)*mineralization_by_ox("c", Cdoc)
            -(64/14)*R_Nitrification
            +(5/4)*(32/14)*R_Denitrification)
    
    
    C_t_dict['Cpy']  = R_Cpy
    C_t_dict['Cpoc'] = R_Cpoc
    C_t_dict['Cpon'] = R_Cpon
    C_t_dict['Cpop'] = R_Cpop
    C_t_dict['Cdoc'] = R_Cdoc
    C_t_dict['Cdon'] = R_Cdon
    C_t_dict['Cdop'] = R_Cdop    
    C_t_dict['Cam']  = R_Cam
    C_t_dict['Cni']  = R_Cni
    C_t_dict['Cph']  = R_Cph
    C_t_dict['Cox']  = R_Cox
    
    return C_t_dict

# =================== Pelagic process rates calculation / =================== #
# =========================================================================== #


def save_C_to_csv(C, sim_start_date, dt, n_iter, output_file_name, boxNo):
    # Generate dates
    start_date = datetime.strptime(sim_start_date, '%Y-%m-%d')
    dates = [start_date + pd.Timedelta(hours=i*dt*24) for i in range(n_iter + 1)]
    # Prepare data for output
    data = {'Date': dates}
    for var, values in C.items():
        data[var] = values
    df = pd.DataFrame(data)
    # Write to file
    df.to_csv(f'{output_file_name}', index=False)


# =========================================================================== #
# =============================== Run CuLPy \ =============================== #
C = simulate_C()

# Save output to csv file
boxNo = 0
output_file_name=f"output_{boxNo}.csv"
save_C_to_csv(C, sim_start_date, dt, n_iter, output_file_name, boxNo)

elapsed_time = time.time() - start_time
print('\nsimulation took: %.2f seconds\n' % elapsed_time)
print('# =================================================================== #')
# =============================== Run CuLPy / =============================== #
# =========================================================================== #
