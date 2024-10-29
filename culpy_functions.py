import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def kmc_reader(file_name):
    kmc_dict = {}
    with open(file_name, "r") as f:
        for line in f:
            line_ = line.split()
            kmc_name = line_[0]
            kmc_values = float(line_[2])
            kmc_dict[kmc_name] = kmc_values
    return kmc_dict

def interpolate_wJDay(sim_start_date, sim_end_date, JDay_start_date, time_step_per_day, file_name):
    dt_minutes = 24 * 60 / time_step_per_day
    inputF_path = f'{file_name}.csv'
    inputF = pd.read_csv(inputF_path)
    start_date = datetime.strptime(JDay_start_date, "%Y-%m-%d")
    inputF['datetime'] = inputF['time'].apply(lambda x: start_date + timedelta(days=x))
    inputF.set_index('datetime', inplace=True)
    new_time_index = pd.date_range(start=sim_start_date, end=sim_end_date, freq=f'{int(dt_minutes)}min')
    interpolated_input = inputF.reindex(inputF.index.union(new_time_index)).interpolate(method='time').loc[new_time_index]
    return interpolated_input

def interpolate_wDate(sim_start_date, sim_end_date, time_step_per_day, file_name):
    dt_minutes = 24 * 60 / time_step_per_day
    inputF_path = f'{file_name}.csv'
    inputF = pd.read_csv(inputF_path)
    inputF['time'] = pd.to_datetime(inputF['time'])
    inputF.set_index('time', inplace=True)
    new_time_index = pd.date_range(start=sim_start_date, end=sim_end_date, freq=f'{int(dt_minutes)}min')
    if isinstance(inputF.index, pd.DatetimeIndex):
        interpolated_input = inputF.reindex(inputF.index.union(new_time_index)).interpolate(method='time').loc[new_time_index]
    else:
        raise ValueError("input does not have a DatetimeIndex, cannot perform time-based interpolation.")
    return interpolated_input

def simulate_C(state_vars_init, n_iter, dt, sediment_option, Q01_Ri, C01_Ri, Q01_BS, C01_BS, Q10_BS, T, V, I_a, salinity, f_day, kmc, H_CL, Altitude):
    # Initialize concentration dictionary
    C1 = {var: np.zeros(n_iter + 1, dtype=np.float64) for var in state_vars_init}
    for var in state_vars_init:
        C1[var][0] = state_vars_init[var]

    # Precompute constants if possible
    dt_np = np.float64(dt)
    H_CL_np = np.float64(H_CL)

    # Start simulation
    for t in tqdm(range(1, n_iter + 1), desc="Simulating"):
        t_minus_one = t - 1

        # Extract previous concentrations
        C1t = {key: C1[key][t_minus_one] for key in C1}

        # Calculate rates
        R1_t = pelagic_process_rates(C1t, T[t_minus_one], V[t_minus_one], H_CL_np, I_a[t_minus_one], salinity[t_minus_one], f_day[t_minus_one], kmc, Altitude)

        # Update concentrations
        for var in state_vars_init:
            influx = (Q01_Ri[t_minus_one] / V[t_minus_one]) * C01_Ri[var][t_minus_one] + \
                     (Q01_BS[t_minus_one] / V[t_minus_one]) * C01_BS[var][t_minus_one]
            outflux = (Q10_BS[t_minus_one] / V[t_minus_one]) * C1[var][t_minus_one]
            C1[var][t] = C1[var][t_minus_one] + (influx - outflux + R1_t[var]) * dt_np

    return C1

def pelagic_process_rates(C_t_dict, T, V, H, I_a, salinity, f_day, kmc, Altitude):
    T, V, H, I_a, salinity, f_day = map(np.array, [T, V, H, I_a, salinity, f_day])

    # Unpack state variables
    Cpy, Cpoc, Cpon, Cpop, Cdoc, Cdon, Cdop, Cam, Cni, Cph, Cox = (
        C_t_dict[var] for var in ["Cpy", "Cpoc", "Cpon", "Cpop", "Cdoc", "Cdon", "Cdop", "Cam", "Cni", "Cph", "Cox"]
    )

    ChlA = (Cpy / kmc['a_C_chl']) * 1000  # Chlorophyll-a biomass in µg/L
    X_I = calculate_light_limitation(ChlA, H, I_a, f_day, kmc)
    X_N = calculate_nutrient_limitation(Cni, Cam, Cph, kmc)
    O2_sat = calculate_O2_saturation(T, salinity, kmc, Altitude)

    # Phytoplankton-Carbon processes
    R_Cpy = calculate_phytoplankton_rates(Cpy, Cox, X_I, X_N, salinity, T, kmc, H)

    # Particulate organic matter processes
    R_Cpoc = calculate_poc_rates(Cpoc, R_Cpy['death'], T, kmc, H)
    R_Cpon = calculate_pon_rates(Cpon, R_Cpy['death'], T, kmc, H)
    R_Cpop = calculate_pop_rates(Cpop, R_Cpy['death'], T, kmc, H)

    # Dissolved organic matter processes
    R_Cdoc = calculate_doc_rates(Cdoc, R_Cpy['excretion'], R_Cpoc['decomposition'], Cox, Cni, T, kmc)
    R_Cdon = calculate_don_rates(Cdon, R_Cpy['excretion'], R_Cpon['decomposition'], Cox, Cni, T, kmc)
    R_Cdop = calculate_dop_rates(Cdop, R_Cpy['excretion'], R_Cpop['decomposition'], Cox, Cni, T, kmc)

    # Nitrogen and oxygen processes
    R_Cam, R_Cni = calculate_nitrogen_rates(Cam, Cni, Cox, R_Cpy['respiration'], R_Cpy['growth'], T, kmc)
    R_Cox = calculate_oxygen_rates(Cox, R_Cpy, R_Cdoc['mineralization'], R_Cni, O2_sat, T, kmc)

    # Phosphate processes
    R_Cph = calculate_phosphate_rates(Cph, R_Cdop['mineralization'], R_Cpy, kmc)

    rates = {
        'Cpy': R_Cpy['net'],
        'Cpoc': R_Cpoc['net'],
        'Cpon': R_Cpon['net'],
        'Cpop': R_Cpop['net'],
        'Cdoc': R_Cdoc['net'],
        'Cdon': R_Cdon['net'],
        'Cdop': R_Cdop['net'],
        'Cam': R_Cam,
        'Cni': R_Cni,
        'Cph': R_Cph,
        'Cox': R_Cox
    }
    
    return rates

# Helper functions
def calculate_phytoplankton_rates(Cpy, Cox, X_I, X_N, salinity, T, kmc, H):
    mortality = kmc['k_mortality'] * kmc['theta_mortality'] ** (T - 20) * Cpy
    salinity_death = kmc['k_salt_death'] * (salinity / (salinity + kmc['K_Sl_salt'])) * Cpy
    growth = kmc['k_growth'] * (kmc['theta_growth'] ** (T - 20)) * (Cox / (Cox + kmc['K_Sl_ox_Cpy'])) * X_I * X_N * Cpy
    respiration = kmc['k_resipration'] * kmc['theta_resipration'] ** (T - 20) * Cpy
    excretion = kmc['k_excration'] * kmc['theta_excration'] ** (T - 20) * Cpy
    settling = (kmc['v_set_Cpy'] / H) * Cpy
    death = mortality + salinity_death
    net = growth - respiration - excretion - death - settling
    return {'net': net, 'growth': growth, 'respiration': respiration, 'excretion': excretion, 'death': death}

def calculate_light_limitation(ChlA, H, I_a, f_day, kmc):
    K_e = kmc['K_be'] + (0.0088 * ChlA) + (0.054 * (ChlA ** (2 / 3)))
    e_const = np.exp(1)
    X_I = (((e_const * f_day) / (K_e * H)) *
           (np.exp((-I_a / kmc['I_s']) * np.exp(-K_e * H)) - np.exp(-I_a / kmc['I_s'])))
    return X_I

def calculate_nutrient_limitation(Cni, Cam, Cph, kmc):
    X_N_N = (Cni + Cam) / (kmc['K_SN'] + Cni + Cam)
    X_N_P = Cph / (kmc['K_SP'] + Cph)
    return X_N_N * X_N_P

def calculate_O2_saturation(T, salinity, kmc, Altitude):
    TKelvin = T + 273.15
    AltEffect = (100 - (0.0035 * 3.28083 * Altitude)) / 100
    ln_stemp = (-139.34411 + (1.575701E+5 / TKelvin) - (6.642308E+7 / (TKelvin ** 2))
                + (1.243800E+10 / (TKelvin ** 3)) - (8.621949E+11 / (TKelvin ** 4)))
    ln_ssalt = (salinity * ((1.7674E-2) - (1.754E+1 / TKelvin) + (2.1407E+3 / (TKelvin ** 2))))
    O2_sat_fresh = np.exp(ln_stemp)
    O2_sat_salt = np.exp(ln_ssalt)
    return AltEffect * (O2_sat_fresh - O2_sat_salt)

def calculate_poc_rates(Cpoc, death, T, kmc, H):
    decomposition = kmc['k_c_decomp'] * (kmc['theta_c_decomp'] ** (T - 20)) * (Cpoc / (Cpoc + kmc['K_Sl_Cpoc_decomp'])) * Cpoc
    settling = (kmc['v_set_Cpoc'] / H) * Cpoc
    net = death - decomposition - settling
    return {'net': net, 'decomposition': decomposition}

def calculate_pon_rates(Cpon, death, T, kmc, H):
    decomposition = kmc['k_n_decomp'] * (kmc['theta_n_decomp'] ** (T - 20)) * (Cpon / (Cpon + kmc['K_Sl_Cpon_decomp'])) * Cpon
    settling = (kmc['v_set_Cpon'] / H) * Cpon
    net = kmc['a_N_C'] * death - decomposition - settling
    return {'net': net, 'decomposition': decomposition}

def calculate_pop_rates(Cpop, death, T, kmc, H):
    decomposition = kmc['k_p_decomp'] * (kmc['theta_p_decomp'] ** (T - 20)) * (Cpop / (Cpop + kmc['K_Sl_Cpop_decomp'])) * Cpop
    settling = (kmc['v_set_Cpop'] / H) * Cpop
    net = kmc['a_P_C'] * death - decomposition - settling
    return {'net': net, 'decomposition': decomposition}

def calculate_doc_rates(Cdoc, excretion, poc_decomposition, Cox, Cni, T, kmc):
    mineralization_ox = mineralization_by_ox("c", Cdoc, Cox, T, kmc)
    mineralization_ni = mineralization_by_ni("c", Cdoc, Cni, Cox, T, kmc)
    net = excretion + poc_decomposition - (mineralization_ox + mineralization_ni)
    return {'net': net, 'mineralization': mineralization_ox + mineralization_ni}

def calculate_don_rates(Cdon, excretion, pon_decomposition, Cox, Cni, T, kmc):
    mineralization_ox = mineralization_by_ox("n", Cdon, Cox, T, kmc)
    mineralization_ni = mineralization_by_ni("n", Cdon, Cni, Cox, T, kmc)
    mineralization = mineralization_ox + mineralization_ni
    net_don = excretion + pon_decomposition - mineralization
    return {'net': net_don, 'mineralization': mineralization}

def calculate_dop_rates(Cdop, excretion, pop_decomposition, Cox, Cni, T, kmc):
    mineralization_ox = mineralization_by_ox("p", Cdop, Cox, T, kmc)
    mineralization_ni = mineralization_by_ni("p", Cdop, Cni, Cox, T, kmc)
    mineralization = mineralization_ox + mineralization_ni
    net_dop = excretion + pop_decomposition - mineralization
    return {'net': net_dop, 'mineralization': mineralization}


def kmc_array_to_dict(kmc_array):
    # This function converts the kmc_array back into a dictionary for use in calculations
    # You'll need to define the keys in the same order as they are in the kmc_array
    kmc_keys = [
        'k_mortality', 'theta_mortality', 'k_salt_death', 'K_Sl_salt',
        'k_growth', 'theta_growth', 'K_Sl_ox_Cpy', 'a_C_chl', 'I_s',
        'k_resipration', 'theta_resipration', 'k_excration', 'theta_excration',
        'v_set_Cpy', 'k_c_decomp', 'theta_c_decomp', 'K_Sl_Cpoc_decomp',
        'v_set_Cpoc', 'k_n_decomp', 'theta_n_decomp', 'K_Sl_Cpon_decomp',
        'v_set_Cpon', 'k_p_decomp', 'theta_p_decomp', 'K_Sl_Cpop_decomp',
        'v_set_Cpop', 'k_c_mnr_ox', 'theta_c_mnr_ox', 'K_Sl_ox_mnr_c',
        'K_Sl_c_mnr_ox', 'k_c_mnr_ni', 'theta_c_mnr_ni', 'K_Si_ox_mnr_c',
        'K_Sl_ni_mnr_c', 'K_Sl_c_mnr_ni', 'k_nitrification', 'theta_nitr',
        'K_Sl_nitr_ox', 'K_Sl_nitr', 'k_denitrification', 'theta_denitr',
        'K_Si_denitr_ox', 'K_Sl_denitr', 'k_raer', 'theta_rear', 'K_be',
        'a_N_C', 'a_P_C', 'a_O2_C', 'K_SN', 'K_SP'
    ]
    kmc = {key: kmc_array[i] for i, key in enumerate(kmc_keys)}
    return kmc
    
def save_C_to_csv(C, sim_start_date, dt, n_iter, output_file_name):
    start_date = datetime.strptime(sim_start_date, '%Y-%m-%d')
    dates = [start_date + timedelta(hours=i * dt * 24) for i in range(n_iter + 1)]
    data = {'Date': dates}
    for var, values in C.items():
        data[var] = values
    df = pd.DataFrame(data)
    df.to_csv(f'{output_file_name}', index=False)

def mineralization_by_ox(var_name, var, Cox, T, kmc):
    return kmc[f'k_{var_name}_mnr_ox'] * (kmc[f'theta_{var_name}_mnr_ox'] ** (T - 20)) * \
           (Cox / (kmc[f'K_Sl_ox_mnr_{var_name}'] + Cox)) * \
           (var / (var + kmc[f'K_Sl_{var_name}_mnr_ox'])) * var

def mineralization_by_ni(var_name, var, Cni, Cox, T, kmc):
    return kmc[f'k_{var_name}_mnr_ni'] * (kmc[f'theta_{var_name}_mnr_ni'] ** (T - 20)) * \
           (1 - Cox / (kmc[f'K_Si_ox_mnr_{var_name}'] + Cox)) * \
           (Cni / (kmc[f'K_Sl_ni_mnr_{var_name}'] + Cni)) * \
           (var / (var + kmc[f'K_Sl_{var_name}_mnr_ni'])) * var

def calculate_nitrogen_rates(Cam, Cni, Cox, respiration, growth, T, kmc):
    prefam = (Cam * (Cni / ((kmc['K_SN'] + Cam) * (kmc['K_SN'] + Cni))) +
              Cam * (kmc['K_SN'] / ((Cam + Cni) * (kmc['K_SN'] + Cni))))
    nitrification = (kmc['k_nitrification'] * (kmc['theta_nitr'] ** (T - 20)) *
                     (Cox / (Cox + kmc['K_Sl_nitr_ox'])) * (Cam / (Cam + kmc['K_Sl_nitr']))) * Cam
    denitrification = (kmc['k_denitrification'] * (kmc['theta_denitr'] ** (T - 20)) *
                       (kmc['K_Si_denitr_ox'] / (Cox + kmc['K_Si_denitr_ox'])) *
                       (Cni / (Cni + kmc['K_Sl_denitr']))) * Cni
    net_ammonia = (nitrification + kmc['a_N_C'] * respiration -
                   kmc['a_N_C'] * prefam * growth - nitrification)
    net_nitrate = nitrification - denitrification - kmc['a_N_C'] * (1 - prefam) * growth
    return net_ammonia, net_nitrate

def calculate_oxygen_rates(Cox, phyto_rates, mineralization_doc, nitrification, denitrification, O2_sat, kmc):
    re_aeration = kmc['k_raer'] * kmc['theta_rear'] * (O2_sat - Cox)
    R_Cox = (re_aeration + kmc['a_O2_C'] * phyto_rates['growth'] -
             kmc['a_O2_C'] * phyto_rates['respiration'] -
             (32 / 12) * mineralization_doc -
             (64 / 14) * nitrification +
             (5 / 4) * (32 / 14) * denitrification)
    return R_Cox

def calculate_phosphate_rates(Cph, dop_mineralization, phyto_rates, kmc):
    return (dop_mineralization + kmc['a_P_C'] * phyto_rates['respiration'] -
            kmc['a_P_C'] * phyto_rates['growth'])

