# This file is part of CuLPy
# Copyright (c) 2024 Burak Kaynaroglu
# This program is free software distributed under the MIT License 
# A copy of the MIT License can be found at 
# https://github.com/kaynarob/CuLPy/blob/main/LICENSE.md
""" CuLPY model example for 0-dimentional configuration - ploting"""



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_all_variables(df_output, df_observation, plot_start_date, plot_end_date):
    # Filter datasets for the specified date range
    df_output = df_output[(df_output.index >= plot_start_date) & (df_output.index <= plot_end_date)]
    df_observation = df_observation[(df_observation.index >= plot_start_date) & (df_observation.index <= plot_end_date)]
    df_output = df_output.resample('5D').mean()

    for variable in df_observation.columns:
        plt.figure(figsize=(12, 6))
        
        # Plotting the simulation values
        plt.plot(df_output.index, df_output[variable], label='Simulation', linestyle='-', color='red')
        
        if variable in df_observation.columns:
            # Plotting the observation values
            plt.scatter(df_observation.index, df_observation[variable], color='blue', label='Observation')

        plt.ylabel(f'{variable} (mg/l)')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        # Rotate the date labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Load the datasets
output_file_name = "output_0.csv"
plot_start_date = "2014-05-01"
plot_end_date = "2023-12-01"
df_output = pd.read_csv(output_file_name, parse_dates=[0], index_col=0)
df_observation = pd.read_csv('input/observation.csv', parse_dates=[0], index_col=0)

plot_all_variables(df_output, df_observation, plot_start_date,  plot_end_date)

