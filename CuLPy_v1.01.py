import time
import numpy as np
import pandas as pd
from datetime import datetime
# from numba import njit, prange

from culpy_functions import (
    kmc_reader,
    interpolate_wJDay,
    interpolate_wDate,
    simulate_C,
    save_C_to_csv
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

C01_Ri = {var: np.array(df_input_C01_Ri[var].values) for var in state_vars_init}
C01_BS = {var: np.array(df_input_C01_BS[var].values) for var in state_vars_init}

T = np.array(df_input_T['tmean'].values)
V = np.array(df_input_V['volume'].values)
I_a = np.array(df_input_Ia['Ia'].values)
f_day = np.array(df_input_fDay['fDay'].values)
salinity = np.array(df_input_Salt['salt'].values)

# Run the simulation
start_time = time.time()
C = simulate_C(
    state_vars_init, n_iter, dt, sediment_option,
    Q01_Ri, C01_Ri, Q01_BS, C01_BS, Q10_BS,
    T, V, I_a, salinity, f_day, kmc, H_CL, Altitude
)
elapsed_time = time.time() - start_time
print(f'\nSimulation completed in {elapsed_time:.2f} seconds\n')

# Save results
save_C_to_csv(C, sim_start_date, dt, n_iter, "output.csv")
elapsed_time = time.time() - start_time
print(f'\nSimulation completed in {elapsed_time:.2f} seconds\n')
