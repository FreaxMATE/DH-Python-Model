"""
Exercise in preparation for Project III
BERN03 course
Lund University
Annemarie Eckes-Shephard

Converted to Python
"""
import sys
sys.path.insert(0, '/home/kunruh/Documents/Studium/Physik/Master/4/ModellingClimateSystem/Projects/3/code/dhmodel-py')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dhmodel import run_dh_model, load_dh_model_ins

# Configuration
# Replace with your own forcing data path:
your_folder = "/home/kunruh/Documents/Studium/Physik/Master/4/ModellingClimateSystem/Projects/3/code/dhmodel-py/data/"
path_forcing = your_folder + "ssp370.csv"
# path_forcing = your_folder + "ssp370.csv"
# path_forcing = your_folder + "ssp585.csv"

# Load forcing data
Forcing = pd.read_csv(path_forcing)

##########################################

# This is how one can subset for particular conditions in the dataframe
M_idx = Forcing['Location'] == "Malmö"
L_idx = Forcing['Location'] == "Luleå"
H_idx = Forcing['Location'] == "Härnösand"

# Example: plot temperature for different locations
# plt.figure(figsize=(12, 6))
# plt.plot(Forcing[M_idx]['Temp'].values, label='Malmö')
# # plt.plot(Forcing[H_idx]['Temp'].values, label='Härnösand', color='red')
# # plt.plot(Forcing[L_idx]['Temp'].values, label='Luleå', color='green')
# plt.legend()
# plt.ylabel('Temperature (°C)')
# plt.show()

# Create site-specific dataframes
Malmö_clim = Forcing[M_idx].copy()
Luleå_clim = Forcing[L_idx].copy()
Härnösand_clim = Forcing[H_idx].copy()

# General subsetting example
idx = Forcing['Year'] >= 2090
print("First rows after 2090:")
print(Forcing[idx].head())
print("\nLast rows after 2090:")
print(Forcing[idx].tail())

# Plot the site climate over a timeperiod of your choosing
# (all years, first years vs last years, policy-relevant time-periods),
# at a resolution of your choosing (aggregated annual, monthly, daily) for your site.
# Think here already of the task Santa has given you. What would be relevant
# timeperiods/resolutions to explore and why?

# For the presentation/assessment, do a quick introduction about the location,
# climate today, how the relevant climate variables will change in the future

# Load model data and parameters
DH_model_ins = load_dh_model_ins()

# Check what parameters the model is sensitive to, and think about why:

##########################################
### BASELINE RUNS:
## 25 years, 53 weeks/year:

from dhmodel.utils import calculate_height_from_diameter

# Species-specific allometric parameters for height calculation
# h = exp(a + alpha * log(diameter_cm))
SPECIES_PARAMS = {
    'abies_alba': {'a': 0.8278, 'alpha': 0.6948},      # European silver fir
    'picea_abies': {'a': -0.1753, 'alpha': 0.9113},     # Norway spruce
    'pseudo_menzii': {'a': 0.1377, 'alpha': 0.8005}    # Douglas fir
}

# Select species for simulation
SELECTED_SPECIES = 'abies_alba'
species_params = SPECIES_PARAMS[SELECTED_SPECIES]
print(f"\nUsing species: {SELECTED_SPECIES}")
print(f"Parameters: a = {species_params['a']}, alpha = {species_params['alpha']}")

# Define starting years for simulations (every 10 years, each running for 25 years)
starting_years = list(range(2015, 2100, 10))
print(f"\nSimulating DH model for starting years: {starting_years}")

print('Best models: ', DH_model_ins['Parameters']['best'])
params = [5.5e+00, 6.19e-01, 9.2e-01, 2.0e-01, 5.0e-03, 2.0e-03, 4.0e+00, 5.0e-01, 6.4e-01, 1.0e+01]

##########################################
### Loop through all three species

for SELECTED_SPECIES in ['abies_alba', 'picea_abies', 'pseudo_menzii']:
    species_params = SPECIES_PARAMS[SELECTED_SPECIES]
    print("\n" + "="*70)
    print(f"SIMULATING SPECIES: {SELECTED_SPECIES.replace('_', ' ').title()}")
    print(f"Parameters: a = {species_params['a']}, alpha = {species_params['alpha']}")
    print("="*70)
    
    # Storage for combined plot
    all_height_data = []
    all_timestamps = []
    all_labels = []

    ##########################################
    ### Malmö - Multiple 25-year periods

    print("\n" + "="*50)
    print("Running model for Malmö...")
    print("="*50)

    # Create colormap for different periods
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(starting_years)))

    # Variable to store the simulation duration based on first tree
    simulation_weeks = None

    for period_idx, start_year in enumerate(starting_years):
        print(f"\n--- Period starting at {start_year} ---")
        
        # For the first simulation, use 25 years to determine when tree reaches 1.5m
        # For subsequent simulations, use the duration determined by the first tree
        if period_idx == 0:
            max_years = 25
        else:
            max_years = int(np.ceil(simulation_weeks / 53))
        
        # Find the indices for this period
        year_mask = (Malmö_clim['Year'] >= start_year) & (Malmö_clim['Year'] < start_year + max_years)
        period_indices = np.where(year_mask)[0]
        
        if len(period_indices) < simulation_weeks if simulation_weeks else max_years*53:
            print(f"Warning: Not enough data for {start_year} (only {len(period_indices)} weeks available)")
            if len(period_indices) == 0:
                continue
        
        # For first tree, take up to 25 years; for others, use determined duration
        if period_idx == 0:
            idx_bl = period_indices[:min(len(period_indices), max_years*53)]
        else:
            idx_bl = period_indices[:min(len(period_indices), simulation_weeks)]
        
        # Run model
        Malmö = run_dh_model(
            Tair=Malmö_clim['Temp'].values[idx_bl],
            Rw=Malmö_clim['W'].values[idx_bl],
            Rw_vol=False,
            Cpool=Malmö_clim['Cpool'].values[idx_bl],
            params=params,
            week=Malmö_clim['Week'].values[idx_bl],
            year=Malmö_clim['Year'].values[idx_bl],
            DH_plot=False  # Disable default plots
        )
        
        # Create proper datetime index with weekly frequency
        timestamps = pd.to_datetime(
            [f"{year}-W{week:02d}-1" for year, week in zip(Malmö['years'].values, Malmö['weeks'].values)],
            format='%Y-W%W-%w'
        )
        
        # Calculate cumulative increment and height
        cumulative_incr = np.cumsum(Malmö['Incr'].values * 2)  # *2 from radius to diameter (mm)
        diameter_cm = cumulative_incr / 10  # Convert mm to cm
        height_m = calculate_height_from_diameter(diameter_cm, a=species_params['a'], alpha=species_params['alpha'])
        
        # Find where height reaches 1.5 meters (for tracking time)
        idx_1_5m = np.where(height_m >= 1.5)[0]
        if len(idx_1_5m) > 0:
            time_to_1_5m = idx_1_5m[0]
            print(f"Tree reaches 1.5m at week {time_to_1_5m}, year {Malmö['years'].values[time_to_1_5m]}")
            
            # If this is the first tree, set the simulation duration for all subsequent trees
            # and cut off the data at 1.5m
            if period_idx == 0:
                simulation_weeks = time_to_1_5m + 1  # +1 to include the week it reaches 1.5m
                print(f"*** Setting simulation duration to {simulation_weeks} weeks for all subsequent trees ***")
                # Cut off first tree data at 1.5m
                height_m = height_m[:simulation_weeks]
                timestamps = timestamps[:simulation_weeks]
            
            # Store data for combined plot
            all_height_data.append(height_m)
            all_timestamps.append(timestamps)
            all_labels.append(f'{start_year}')
        else:
            # Tree doesn't reach 1.5m in available data
            time_to_1_5m = None
            print(f"Tree does not reach 1.5m in the simulation period (max: {height_m[-1]:.2f}m)")
            
            # Check if we ran out of data (simulation period extends beyond 2100)
            if len(idx_bl) < (simulation_weeks if simulation_weeks else max_years*53):
                print(f"Skipping tree starting in {start_year} - insufficient data and did not reach 1.5m")
            else:
                print(f"Warning: Tree starting in {start_year} did not reach 1.5m despite having full simulation period")
                # Still include it in the plot for completeness
                all_height_data.append(height_m)
                all_timestamps.append(timestamps)
                all_labels.append(f'{start_year}')

    # Create combined plot with all tree height time series for this species
    print("\n" + "="*50)
    print(f"Creating combined tree height plot for {SELECTED_SPECIES}...")
    print("="*50)

    plt.figure(figsize=(14, 8))
    for idx, (heights, times, label) in enumerate(zip(all_height_data, all_timestamps, all_labels)):
        color = colors_map[idx]
        plt.plot(times, heights, linewidth=2, color=color, label=label, alpha=0.8)

    plt.axhline(y=1.5, color='r', linestyle='--', linewidth=2, label='1.5 meters threshold')
    plt.ylabel('Tree Height (m)')
    plt.xlabel('Date')
    plt.title(f'Malmö - {SELECTED_SPECIES.replace("_", " ").title()} Height Time Series Including Growth Beyond 1.5m (2015-2100)')
    plt.legend(title='Starting Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save figure
    filename = f'malmo_{SELECTED_SPECIES}_height_timeseries.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.show()

print("\n" + "="*50)
print("All Malmö simulations complete!")
print("="*50)
