import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from numba import njit

from numba_functions import (
    kmc_reader,
    interpolate_wJDay,
    interpolate_wDate,
    save_C_to_csv, simulate_C
)

# Constants
H_CL = 3.2751               # Average water depth
Altitude = 0.0              # Site specific altitude (m).

# Initialize and set up simulation parameters
sim_start_date = "2014-01-01"
sim_end_date = "2024-12-31"
JDay_start_date = "2014-01-01"
dt = 1 / 240  # timestep in days
kmc_file_name = 'input/constants_pelagic_0d.txt'
state_vars_init = {
    "Cpy": 0.680, "Cpoc": 2.000, "Cpon": 0.270, "Cpop": 0.045,
    "Cdoc": 11.00, "Cdon": 4.270, "Cdop": 0.005,
    "Cam": 0.040, "Cni": 2.769, "Cph": 0.022, "Cox": 13.33
}

sediment_option = 0

print('\n# =================================================================== #')
print('\nmodel initialization...\n')
total_boxNo = 1
print(f'\tnumber of boxes                : {total_boxNo}')
sediment_option = 0
print(f'\tsediment_option                : {int(sediment_option)}')
print(f'\tpelagic constants file name    : {kmc_file_name}')
print(f'\tsimulation start date          : {sim_start_date}')
print(f'\tsimulation end date            : {sim_end_date}')
print(f'\ttime step in days              : {int(1/dt)}')
print(f'\tdt                             : {int(24*60*dt)} minute(s)')

# Calculate the number of iterations
date_1 = datetime.strptime(sim_start_date, '%Y-%m-%d')
date_2 = datetime.strptime(sim_end_date, '%Y-%m-%d')
sim_end_jdays = (date_2 - date_1).days
n_iter = int(sim_end_jdays / dt)

# Load constants and input data
kmc = kmc_reader(kmc_file_name)
df_input_Q = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "input/fluxes_0d")
df_input_T = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "input/temperature")
df_input_V = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "input/volume_t_0d")
df_input_Ia = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "input/solar_radiation")
df_input_fDay = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "input/fraction_daylight")
df_input_Salt = interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, 1/dt, "input/salinity")
df_input_C01_BS = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, "input/bc_concentration_0")
df_input_C01_Ri = interpolate_wDate(sim_start_date, sim_end_date, 1/dt, "input/bc_concentration_1")

# Convert data to NumPy arrays and set up other parameters
Q01_Ri = np.array(df_input_Q['q01_Riv'].values * 86400)  # Flow rates in m³/day
Q01_BS = np.array(df_input_Q['q01_BS'].values * 86400)
Q10_BS = np.array(df_input_Q['q10_BS'].values * 86400)

# Convert concentration inputs to arrays, extracting only the keys defined in state_vars_init
state_vars_keys = list(state_vars_init.keys())
state_vars_init_array = np.array([state_vars_init[key] for key in state_vars_keys])
C01_Ri_array = np.array([df_input_C01_Ri[key].values for key in state_vars_keys])
C01_BS_array = np.array([df_input_C01_BS[key].values for key in state_vars_keys])

T = np.array(df_input_T['tmean'].values)
V = np.array(df_input_V['volume'].values)
I_a = np.array(df_input_Ia['Ia'].values)
f_day = np.array(df_input_fDay['fDay'].values)
salinity = np.array(df_input_Salt['salt'].values)

kmc_keys = [
    'k_growth', 'k_resipration', 'k_mortality', 'k_excration', 'k_salt_death',  
    'v_set_Cpy', 'K_SN', 'K_SP', 'K_Sl_salt', 'K_Sl_ox_Cpy', 'k_c_decomp', 
    'k_n_decomp', 'k_p_decomp', 'v_set_Cpoc', 'v_set_Cpon', 'v_set_Cpop',
    'k_c_mnr_ox', 'k_c_mnr_ni', 'k_n_mnr_ox', 'k_n_mnr_ni', 'k_p_mnr_ox', 
    'k_p_mnr_ni', 'K_Sl_Cpoc_decomp', 'K_Sl_Cpon_decomp', 'K_Sl_Cpop_decomp', 
    'K_Sl_c_mnr_ox', 'K_Sl_c_mnr_ni', 'K_Sl_ox_mnr_c', 'K_Si_ox_mnr_c', 
    'K_Sl_ni_mnr_c', 'K_Sl_n_mnr_ox', 'K_Sl_n_mnr_ni', 'K_Sl_ox_mnr_n', 
    'K_Si_ox_mnr_n', 'K_Sl_ni_mnr_n', 'K_Sl_p_mnr_ox', 'K_Sl_p_mnr_ni', 
    'K_Sl_ox_mnr_p', 'K_Si_ox_mnr_p', 'K_Sl_ni_mnr_p', 'k_nitrification', 
    'K_Sl_nitr', 'K_Sl_nitr_ox', 'k_denitrification', 'K_Sl_denitr', 
    'K_Si_denitr_ox', 'k_raer', 'K_be', 'I_s', 'theta_growth', 
    'theta_resipration', 'theta_mortality', 'theta_excration', 'theta_c_decomp', 
    'theta_n_decomp', 'theta_p_decomp', 'theta_c_mnr_ox', 'theta_n_mnr_ox', 
    'theta_p_mnr_ox', 'theta_c_mnr_ni', 'theta_n_mnr_ni', 'theta_p_mnr_ni', 
    'theta_nitr', 'theta_denitr', 'theta_rear', 'a_C_chl', 'a_N_C', 'a_P_C', 
    'a_O2_C'
]

# Convert kmc values to array for Numba compatibility
kmc_array = np.array([kmc[key] for key in kmc_keys])

# Run the simulation with updated arguments for `simulate_C`
start_time = time.time()
C_array = simulate_C(
    state_vars_init_array, n_iter, dt, sediment_option,
    Q01_Ri, C01_Ri_array, Q01_BS, C01_BS_array, Q10_BS,
    T, V, I_a, salinity, f_day, kmc_array, H_CL, Altitude
)
elapsed_time = time.time() - start_time
print(f'\nSimulation completed in {elapsed_time:.2f} seconds\n')

# Convert `C_array` back to dictionary format for saving to CSV
C = {var: C_array[i, :] for i, var in enumerate(state_vars_keys)}
save_C_to_csv(C, sim_start_date, dt, n_iter, "output.csv")
elapsed_time = time.time() - start_time
print(f'\nSimulation completed in {elapsed_time:.2f} seconds\n')
