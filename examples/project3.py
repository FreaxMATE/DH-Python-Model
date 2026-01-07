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
from scipy import stats

# Configuration
# Replace with your own forcing data path:
your_folder = "/home/kunruh/Documents/Studium/Physik/Master/4/ModellingClimateSystem/Projects/3/code/dhmodel-py/data/"

# ==================================================
# USER CONFIGURATION: Select scenarios and species
# ==================================================

# Select which scenarios to analyze (comment out to exclude)
SCENARIOS_TO_RUN = [
    'ssp126',
    'ssp245',
    'ssp370',
    'ssp585'
]

# Select which species to analyze (comment out to exclude)
SPECIES_TO_RUN = [
    'abies_alba',      # European silver fir
    'picea_abies',     # Norway spruce
    'pseudo_menzii'    # Douglas fir
]

print(f"\n{'='*70}")
print(f"CONFIGURATION:")
print(f"  Scenarios to run: {', '.join([s.upper() for s in SCENARIOS_TO_RUN])}")
print(f"  Species to run: {', '.join([s.replace('_', ' ').title() for s in SPECIES_TO_RUN])}")
print(f"{'='*70}\n")

# All available scenarios (for reference)
scenarios = SCENARIOS_TO_RUN

##########################################

# Load model data and parameters
DH_model_ins = load_dh_model_ins()

print(DH_model_ins)

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

# Define starting points for simulations (every 2 weeks from 2015 to 2100)
# Create list of (year, week) tuples for starting points
starting_points = []
for year in range(2015, 2100):
    for week in range(1, 53, 2):  # Every 2 weeks (weeks 1, 3, 5, 7, ...)
        starting_points.append((year, week))
print(f"\nSimulating DH model with {len(starting_points)} starting points (every 2 weeks from 2015-2100)")

print('Best models: ', DH_model_ins['Parameters']['best'])
params = [5.5e+00, 6.19e-01, 9.2e-01, 2.0e-01, 5.0e-03, 2.0e-03, 4.0e+00, 5.0e-01, 6.4e-01, 1.0e+01]

##########################################
### Plot 20-year moving averages of temperature and soil moisture for all scenarios

print("\n" + "="*70)
print("Creating 20-year moving average plots for temperature and soil moisture...")
print("="*70)

# Define colors for each scenario (same as used elsewhere in script)
scenario_colors = {
    'ssp126': '#1f77b4',  # Blue
    'ssp245': '#2ca02c',  # Green
    'ssp370': '#ff7f0e',  # Orange
    'ssp585': '#d62728'   # Red
}

# Moving average window (20 years)
ma_window_years = 20

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

for scenario in scenarios:
    # Load forcing data for this scenario
    path_forcing = your_folder + f"{scenario}.csv"
    Forcing = pd.read_csv(path_forcing)
    
    # Filter for Malmö
    Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
    
    # Calculate annual mean temperature and soil moisture
    annual_temp = Malmö_data.groupby('Year')['Temp'].mean()
    annual_W = Malmö_data.groupby('Year')['W'].mean()
    
    # Calculate 20-year moving averages
    ma_temp = annual_temp.rolling(window=ma_window_years, center=True, min_periods=1).mean()
    ma_W = annual_W.rolling(window=ma_window_years, center=True, min_periods=1).mean()
    
    # Get color for this scenario
    color = scenario_colors.get(scenario, '#000000')
    
    # Plot temperature
    ax1.plot(annual_temp.index, annual_temp.values, linewidth=0.5, color=color, alpha=0.2)
    ax1.plot(ma_temp.index, ma_temp.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)
    
    # Plot soil moisture
    ax2.plot(annual_W.index, annual_W.values, linewidth=0.5, color=color, alpha=0.2)
    ax2.plot(ma_W.index, ma_W.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)

# Format temperature plot
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_title(f'Annual Mean Temperature ({ma_window_years}-Year Moving Average)\nMalmö', fontsize=12, fontweight='bold')
ax1.legend(title='Scenario', fontsize=9)
ax1.grid(True, alpha=0.3)

# Format soil moisture plot
ax2.set_ylabel('Soil Moisture (W)', fontsize=11)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_title(f'Annual Mean Soil Moisture ({ma_window_years}-Year Moving Average)\nMalmö', fontsize=12, fontweight='bold')
ax2.legend(title='Scenario', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
filename = 'malmo_20yr_moving_averages_temp_soilmoisture.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved figure: {filename}")
plt.show()

##########################################
### Loop through all scenarios and species

# Storage for growth rates
growth_rate_data = {scenario: [] for scenario in scenarios}

for scenario in scenarios:
    print("\n" + "#"*70)
    print(f"### SCENARIO: {scenario.upper()} ###")
    print("#"*70)
    
    # Load forcing data for this scenario
    path_forcing = your_folder + f"{scenario}.csv"
    Forcing = pd.read_csv(path_forcing)
    
    # Create site-specific dataframes
    M_idx = Forcing['Location'] == "Malmö"
    L_idx = Forcing['Location'] == "Luleå"
    H_idx = Forcing['Location'] == "Härnösand"
    
    Malmö_clim = Forcing[M_idx].copy()
    Luleå_clim = Forcing[L_idx].copy()
    Härnösand_clim = Forcing[H_idx].copy()
    
    for SELECTED_SPECIES in SPECIES_TO_RUN:
        species_params = SPECIES_PARAMS[SELECTED_SPECIES]
        print("\n" + "="*70)
        print(f"SIMULATING SPECIES: {SELECTED_SPECIES.replace('_', ' ').title()} - Scenario: {scenario.upper()}")
        print(f"Parameters: a = {species_params['a']}, alpha = {species_params['alpha']}")
        print("="*70)
        
        # Storage for combined plot and growth rates
        all_height_data = []
        all_timestamps = []
        all_labels = []
        all_time_years = []
        all_fits = []
        all_time_to_1_5m = []  # Store time to reach 1.5m for each plot
        # Storage for ALL simulations (for complete plot)
        all_height_data_complete = []
        all_time_years_complete = []
        all_fits_complete = []
        all_start_years_complete = []
        scenario_growth_rates = []
        scenario_residual_errors = []
        scenario_years = []
        skipped_count = 0

        ##########################################
        ### Malmö - Multiple 25-year periods

        print("\n" + "="*50)
        print("Running model for Malmö...")
        print("="*50)

        # Create colormap for different periods (only for plotting every 52 weeks = 1 year)
        num_years = (2100 - 2015)
        colors_map = plt.cm.viridis(np.linspace(0, 1, num_years))

        # Maximum simulation duration (25 years)
        max_simulation_weeks = 25 * 53

        for period_idx, (start_year, start_week) in enumerate(starting_points):
            if period_idx % 13 == 0:  # Print progress every 13 simulations (1 year)
                print(f"--- Year {start_year} (simulation {period_idx + 1}/{len(starting_points)}) ---")
            
            # Find the starting index for this period (specific year and week)
            start_mask = (Malmö_clim['Year'] == start_year) & (Malmö_clim['Week'] == start_week)
            start_indices = np.where(start_mask)[0]
            
            if len(start_indices) == 0:
                skipped_count += 1
                continue
                
            start_idx = start_indices[0]
            
            # Run simulation for as long as data is available (up to 25 years)
            end_idx = min(start_idx + max_simulation_weeks, len(Malmö_clim))
            
            # Skip if we don't have at least 1 year of data
            if end_idx - start_idx < 53:
                skipped_count += 1
                continue
            
            idx_bl = np.arange(start_idx, end_idx)
            
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
            # Add small minimum diameter to avoid log(0) warnings
            diameter_cm = np.maximum(diameter_cm, 1e-6)  # Minimum 0.000001 cm
            height_m = calculate_height_from_diameter(diameter_cm, a=species_params['a'], alpha=species_params['alpha'])
            
            # Find where height reaches 1.5 meters (for tracking time)
            idx_1_5m = np.where(height_m >= 1.5)[0]
            if len(idx_1_5m) > 0:
                time_to_1_5m = idx_1_5m[0]
                
                # Cut off the data at 1.5m for all trees
                height_m = height_m[:time_to_1_5m + 1]  # +1 to include the week it reaches 1.5m
                timestamps = timestamps[:time_to_1_5m + 1]
                
                # Calculate growth rate using linear fit
                # Convert timestamps to years for fitting
                time_years = np.array([(t - timestamps[0]).total_seconds() / (365.25 * 24 * 3600) for t in timestamps])
                
                # Linear fit: height = slope * time + intercept
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_years, height_m)
                
                # Calculate residual standard error
                y_pred = slope * time_years + intercept
                residuals = height_m - y_pred
                residual_std_error = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))  # RSS / (n-2)
                
                # Store growth rate (slope = m/year) and residual standard error
                scenario_growth_rates.append(slope)
                scenario_residual_errors.append(residual_std_error)
                # Convert to decimal year for plotting
                decimal_year = start_year + (start_week - 1) / 52.0
                scenario_years.append(decimal_year)
                
                # Store ALL simulations for complete plot
                all_height_data_complete.append(height_m)
                all_time_years_complete.append(time_years)
                all_fits_complete.append({'slope': slope, 'intercept': intercept, 'r_value': r_value})
                all_start_years_complete.append(decimal_year)  # Use decimal year for accurate plotting
                
                # Store data for combined plot (only every 6 years to keep plots readable)
                # 6 years * 13 simulations per year = 78 simulations
                if period_idx % 78 == 0:
                    all_height_data.append(height_m)
                    all_timestamps.append(timestamps)
                    all_labels.append(f'{start_year}')
                    all_time_years.append(time_years)
                    all_fits.append({'slope': slope, 'intercept': intercept, 'r_value': r_value})
                    # Store time to 1.5m (in years)
                    time_to_reach_1_5m = time_years[-1]  # Last time value when it reaches 1.5m
                    all_time_to_1_5m.append(time_to_reach_1_5m)
            else:
                # Tree doesn't reach 1.5m in available data - skip this simulation
                skipped_count += 1
        
        print(f"\n*** Completed simulations: {len(scenario_growth_rates)}/{len(starting_points)} (skipped: {skipped_count}) ***\n")
        
        # Store growth rates for this species and scenario
        growth_rate_data[scenario].append({
            'species': SELECTED_SPECIES,
            'years': scenario_years,
            'growth_rates': scenario_growth_rates,
            'residual_errors': scenario_residual_errors
        })

        # Create combined plot with all tree height time series for this species
        print("\n" + "="*50)
        print(f"Creating combined tree height plot for {SELECTED_SPECIES}...")
        print("="*50)

        # Only plot 10 starting years evenly distributed across the range
        if len(all_height_data) > 10:
            # Select 10 evenly spaced indices
            indices_to_plot = np.linspace(0, len(all_height_data) - 1, 10, dtype=int)
            heights_to_plot = [all_height_data[i] for i in indices_to_plot]
            times_to_plot = [all_timestamps[i] for i in indices_to_plot]
            labels_to_plot = [all_labels[i] for i in indices_to_plot]
            time_to_1_5m_plot = [all_time_to_1_5m[i] for i in indices_to_plot]
            time_years_plot = [all_time_years[i] for i in indices_to_plot]
        else:
            heights_to_plot = all_height_data
            times_to_plot = all_timestamps
            labels_to_plot = all_labels
            time_to_1_5m_plot = all_time_to_1_5m
            time_years_plot = all_time_years

        plt.figure(figsize=(20, 10))
        
        # Create a colormap for time gradient (years since start)
        time_cmap = plt.cm.viridis
        
        for idx, (heights, times, label, time_to_target, time_years) in enumerate(zip(heights_to_plot, times_to_plot, labels_to_plot, time_to_1_5m_plot, time_years_plot)):
            # Underlay with light gray to show the full growth period
            plt.fill_between(times, 0, heights, color='lightgray', alpha=0.3, linewidth=0)
            
            # Draw connecting lines
            plt.plot(times, heights, color='black', linewidth=1.5, alpha=0.3, zorder=1)
            
            # Create scatter plot with color gradient based on years since start
            scatter = plt.scatter(times, heights, c=time_years, cmap=time_cmap, 
                                s=30, alpha=0.8, vmin=0, vmax=max([max(ty) for ty in time_years_plot]),
                                edgecolors='none', zorder=2)
            
            # Add label showing time to 1.5m (double size and bold)
            # Place label near the end of the growth curve
            label_x = times[-1]
            label_y = heights[-1]
            plt.text(label_x, label_y, f'{time_to_target:.1f}y', 
                    fontsize=18, fontweight='bold', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

        # Add colorbar for time gradient
        cbar = plt.colorbar(scatter, ax=plt.gca(), label='Years since planting')
        
        plt.axhline(y=1.5, color='r', linestyle='--', linewidth=2, label='1.5 meters threshold')
        plt.ylabel('Tree Height (m)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.title(f'Malmö - {SELECTED_SPECIES.replace("_", " ").title()} Height Time Series - {scenario.upper()} (10 Starting Years)', fontsize=13, fontweight='bold')
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        # Save figure
        filename = f'malmo_{SELECTED_SPECIES}_{scenario}_height_timeseries.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close()  # Close figure to save memory
        
        # Create combined plot with height data and linear fits
        print(f"Creating combined height and fit plot for {SELECTED_SPECIES}...")
        
        # Only plot 10 starting years evenly distributed across the range
        if len(all_height_data) > 10:
            # Select 10 evenly spaced indices (same as for timeseries plot)
            indices_to_plot = np.linspace(0, len(all_height_data) - 1, 10, dtype=int)
            heights_to_plot = [all_height_data[i] for i in indices_to_plot]
            times_to_plot = [all_time_years[i] for i in indices_to_plot]
            labels_to_plot = [all_labels[i] for i in indices_to_plot]
            fits_to_plot = [all_fits[i] for i in indices_to_plot]
        else:
            heights_to_plot = all_height_data
            times_to_plot = all_time_years
            labels_to_plot = all_labels
            fits_to_plot = all_fits
        
        plt.figure(figsize=(14, 8))
        for idx, (heights, time_years, label, fit_params) in enumerate(zip(heights_to_plot, times_to_plot, labels_to_plot, fits_to_plot)):
            color_idx = min(int(label) - 2015, len(colors_map) - 1)
            color = colors_map[color_idx]
            
            # Plot actual height data
            plt.plot(time_years, heights, 'o-', linewidth=1.5, markersize=3, color=color, alpha=0.6, label=f'{label}')
            
            # Plot linear fit
            fit_line = fit_params['slope'] * time_years + fit_params['intercept']
            plt.plot(time_years, fit_line, '--', linewidth=2, color=color, alpha=0.8)
        
        plt.axhline(y=1.5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='1.5m threshold')
        plt.xlabel('Time since planting (years)', fontsize=12)
        plt.ylabel('Tree Height (m)', fontsize=12)
        plt.title(f'Malmö - {SELECTED_SPECIES.replace("_", " ").title()} Height with Linear Fits - {scenario.upper()}\n(10 Starting Years - Solid: simulated data, Dashed: linear fits)', 
                 fontsize=13, fontweight='bold')
        plt.legend(title='Starting Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        fit_filename = f'malmo_{SELECTED_SPECIES}_{scenario}_height_fits.png'
        plt.savefig(fit_filename, dpi=300, bbox_inches='tight')
        print(f"Saved combined fit figure: {fit_filename}")
        plt.close()  # Close figure to save memory
        
        # Create plot with ALL trees and their fits
        print(f"Creating complete height and fit plot for ALL simulations ({len(all_height_data_complete)} trees)...")
        
        plt.figure(figsize=(14, 8))
        
        # Plot all trees with low alpha to show density
        for idx, (heights, time_years, fit_params, start_decimal_year) in enumerate(zip(all_height_data_complete, all_time_years_complete, all_fits_complete, all_start_years_complete)):
            # Convert time since planting to absolute years
            absolute_years = start_decimal_year + time_years
            
            # Color based on starting year
            color_idx = min(int(start_decimal_year) - 2015, len(colors_map) - 1)
            color = colors_map[color_idx]
            
            # Plot actual height data with very low alpha
            plt.plot(absolute_years, heights, '-', linewidth=0.5, color=color, alpha=0.15)
            
            # Plot linear fit with slightly higher alpha
            fit_line = fit_params['slope'] * time_years + fit_params['intercept']
            plt.plot(absolute_years, fit_line, '--', linewidth=0.8, color=color, alpha=0.25)
        
        plt.axhline(y=1.5, color='red', linestyle=':', linewidth=2, alpha=0.7, label='1.5m threshold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Tree Height (m)', fontsize=12)
        plt.title(f'Malmö - {SELECTED_SPECIES.replace("_", " ").title()} ALL Simulations ({len(all_height_data_complete)} trees) - {scenario.upper()}\n(Thin lines: simulated data, Dashed: linear fits)', 
                 fontsize=13, fontweight='bold')
        
        # Add colorbar to show year gradient
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=2015, vmax=2099))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Planting Year')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        complete_filename = f'malmo_{SELECTED_SPECIES}_{scenario}_height_fits_complete.png'
        plt.savefig(complete_filename, dpi=300, bbox_inches='tight')
        print(f"Saved complete fit figure: {complete_filename}")
        plt.close()  # Close figure to save memory

print("\n" + "="*50)
print("All simulations complete!")
print("="*50)

##########################################
### Create cumulative threshold plots for temperature and soil moisture

print("\n" + "="*70)
print("Creating cumulative threshold exceedance plots...")
print("="*70)

# Define colors for each scenario
scenario_colors = {
    'ssp126': '#1f77b4',  # Blue
    'ssp245': '#2ca02c',  # Green
    'ssp370': '#ff7f0e',  # Orange
    'ssp585': '#d62728'   # Red
}

# Extract threshold values from params
Tmin = params[0]  # Temperature threshold (5.5°C)
Rwmin = params[1]  # Soil moisture threshold (0.619)

print(f"Temperature threshold (Tmin): {Tmin}°C")
print(f"Soil moisture threshold (Rwmin): {Rwmin}")


##########################################
### Create cumulative threshold plots for temperature and soil moisture

print("\n" + "="*70)
print("Creating cumulative threshold exceedance plots...")
print("="*70)

# Create plots for each threshold with both cumulative and annual counts
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))

# Moving average window (20 years)
ma_window_years = 20

for scenario in SCENARIOS_TO_RUN:
    # Load forcing data for this scenario
    path_forcing = your_folder + f"{scenario}.csv"
    Forcing = pd.read_csv(path_forcing)
    
    # Filter for Malmö
    Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
    
    # Calculate where conditions are below threshold
    temp_below = (Malmö_data['Temp'] < Tmin).astype(int)
    moisture_below = (Malmö_data['W'] < Rwmin).astype(int)
    
    # Calculate cumulative counts
    cumulative_temp = temp_below.cumsum()
    cumulative_moisture = moisture_below.cumsum()
    
    # Calculate annual counts
    Malmö_data['temp_below'] = temp_below
    Malmö_data['moisture_below'] = moisture_below
    annual_temp = Malmö_data.groupby('Year')['temp_below'].sum()
    annual_moisture = Malmö_data.groupby('Year')['moisture_below'].sum()
    
    # Calculate 20-year moving averages
    ma_temp = annual_temp.rolling(window=ma_window_years, center=True, min_periods=1).mean()
    ma_moisture = annual_moisture.rolling(window=ma_window_years, center=True, min_periods=1).mean()
    
    # Create datetime for x-axis
    dates = pd.to_datetime(
        [f"{year}-W{week:02d}-1" for year, week in zip(Malmö_data['Year'].values, Malmö_data['Week'].values)],
        format='%Y-W%W-%w'
    )
    
    # Get color for this scenario
    color = scenario_colors.get(scenario, '#000000')
    
    # Plot cumulative temperature threshold exceedances
    ax1.plot(dates, cumulative_temp, linewidth=2, color=color, label=scenario.upper(), alpha=0.8)
    
    # Plot cumulative soil moisture threshold exceedances
    ax2.plot(dates, cumulative_moisture, linewidth=2, color=color, label=scenario.upper(), alpha=0.8)
    
    # Plot annual temperature exceedances (transparent)
    ax3.plot(annual_temp.index, annual_temp.values, linewidth=1, marker='o', markersize=2, color=color, alpha=0.2)
    # Plot 20-year moving average on top
    ax3.plot(ma_temp.index, ma_temp.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)
    
    # Plot annual soil moisture exceedances (transparent)
    ax4.plot(annual_moisture.index, annual_moisture.values, linewidth=1, marker='o', markersize=2, color=color, alpha=0.2)
    # Plot 20-year moving average on top
    ax4.plot(ma_moisture.index, ma_moisture.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)

# Format cumulative temperature plot
ax1.set_ylabel('Cumulative Weeks', fontsize=11)
ax1.set_title(f'Cumulative Weeks with Temperature < {Tmin}°C\nMalmö', fontsize=12, fontweight='bold')
ax1.legend(title='Scenario', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelbottom=False)

# Format cumulative soil moisture plot
ax2.set_ylabel('Cumulative Weeks', fontsize=11)
ax2.set_title(f'Cumulative Weeks with Soil Moisture < {Rwmin}\nMalmö', fontsize=12, fontweight='bold')
ax2.legend(title='Scenario', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelbottom=False)

# Format annual temperature plot
ax3.set_ylabel('Weeks per Year', fontsize=11)
ax3.set_xlabel('Year', fontsize=11)
ax3.set_title(f'Annual Weeks with Temperature < {Tmin}°C ({ma_window_years}-Year Moving Average)\nMalmö', fontsize=12, fontweight='bold')
ax3.legend(title='Scenario', fontsize=9)
ax3.grid(True, alpha=0.3)

# Format annual soil moisture plot
ax4.set_ylabel('Weeks per Year', fontsize=11)
ax4.set_xlabel('Year', fontsize=11)
ax4.set_title(f'Annual Weeks with Soil Moisture < {Rwmin} ({ma_window_years}-Year Moving Average)\nMalmö', fontsize=12, fontweight='bold')
ax4.legend(title='Scenario', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
filename = 'malmo_cumulative_threshold_exceedances.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved cumulative and annual threshold plot: {filename}")
plt.show()

# Also create a combined plot showing both cumulative and annual counts
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))

for scenario in SCENARIOS_TO_RUN:
    # Load forcing data for this scenario
    path_forcing = your_folder + f"{scenario}.csv"
    Forcing = pd.read_csv(path_forcing)
    
    # Filter for Malmö
    Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
    
    # Calculate where both conditions are below threshold (limiting growth)
    both_below = ((Malmö_data['Temp'] < Tmin) | (Malmö_data['W'] < Rwmin)).astype(int)
    
    # Calculate cumulative counts
    cumulative_both = both_below.cumsum()
    
    # Calculate annual counts
    Malmö_data['both_below'] = both_below
    annual_both = Malmö_data.groupby('Year')['both_below'].sum()
    
    # Calculate 20-year moving average
    ma_both = annual_both.rolling(window=ma_window_years, center=True, min_periods=1).mean()
    
    # Create datetime for x-axis
    dates = pd.to_datetime(
        [f"{year}-W{week:02d}-1" for year, week in zip(Malmö_data['Year'].values, Malmö_data['Week'].values)],
        format='%Y-W%W-%w'
    )
    
    # Get color for this scenario
    color = scenario_colors.get(scenario, '#000000')
    
    # Plot cumulative combined threshold exceedances
    ax_left.plot(dates, cumulative_both, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.8)
    
    # Plot annual combined threshold exceedances (transparent)
    ax_right.plot(annual_both.index, annual_both.values, linewidth=1, marker='o', markersize=2, color=color, alpha=0.2)
    # Plot 20-year moving average on top
    ax_right.plot(ma_both.index, ma_both.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)

# Format left plot (cumulative)
ax_left.set_ylabel('Cumulative Weeks', fontsize=12)
ax_left.set_xlabel('Year', fontsize=12)
ax_left.set_title(f'Cumulative Weeks with Growth Limitation\n(Temperature < {Tmin}°C OR Soil Moisture < {Rwmin}) - Malmö', 
            fontsize=12, fontweight='bold')
ax_left.legend(title='Scenario', fontsize=10)
ax_left.grid(True, alpha=0.3)

# Format right plot (annual counts)
ax_right.set_ylabel('Weeks per Year', fontsize=12)
ax_right.set_xlabel('Year', fontsize=12)
ax_right.set_title(f'Annual Weeks with Growth Limitation ({ma_window_years}-Year Moving Average)\n(Temperature < {Tmin}°C OR Soil Moisture < {Rwmin}) - Malmö', 
            fontsize=12, fontweight='bold')
ax_right.legend(title='Scenario', fontsize=10)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
filename = 'malmo_cumulative_growth_limitation.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved combined threshold plot (cumulative and annual): {filename}")
plt.show()

##########################################
### Create species-specific threshold exceedance plots

print("\n" + "="*70)
print("Creating species-specific threshold exceedance plots...")
print("="*70)

# Create individual plots for each species
for species in SPECIES_TO_RUN:
    species_label = species.replace('_', ' ').title()
    
    fig, ((ax_temp_cum, ax_moisture_cum), (ax_temp_annual, ax_moisture_annual)) = plt.subplots(2, 2, figsize=(18, 12))
    
    for scenario in SCENARIOS_TO_RUN:
        # Load forcing data for this scenario
        path_forcing = your_folder + f"{scenario}.csv"
        Forcing = pd.read_csv(path_forcing)
        
        # Filter for Malmö
        Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
        
        # Calculate where temperature is below threshold
        temp_below = (Malmö_data['Temp'] < Tmin).astype(int)
        
        # Calculate where soil moisture is below threshold
        moisture_below = (Malmö_data['W'] < Rwmin).astype(int)
        
        # Calculate cumulative counts
        cumulative_temp = temp_below.cumsum()
        cumulative_moisture = moisture_below.cumsum()
        
        # Calculate annual counts
        Malmö_data['temp_below'] = temp_below
        Malmö_data['moisture_below'] = moisture_below
        annual_temp = Malmö_data.groupby('Year')['temp_below'].sum()
        annual_moisture = Malmö_data.groupby('Year')['moisture_below'].sum()
        
        # Calculate 20-year moving averages
        ma_temp = annual_temp.rolling(window=ma_window_years, center=True, min_periods=1).mean()
        ma_moisture = annual_moisture.rolling(window=ma_window_years, center=True, min_periods=1).mean()
        
        # Create datetime for x-axis
        dates = pd.to_datetime(
            [f"{year}-W{week:02d}-1" for year, week in zip(Malmö_data['Year'].values, Malmö_data['Week'].values)],
            format='%Y-W%W-%w'
        )
        
        # Get color for this scenario
        color = scenario_colors.get(scenario, '#000000')
        
        # Plot cumulative temperature exceedances
        ax_temp_cum.plot(dates, cumulative_temp, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.8)
        
        # Plot cumulative moisture exceedances
        ax_moisture_cum.plot(dates, cumulative_moisture, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.8)
        
        # Plot annual temperature exceedances (transparent)
        ax_temp_annual.plot(annual_temp.index, annual_temp.values, linewidth=1, marker='o', markersize=2, color=color, alpha=0.2)
        # Plot 20-year moving average on top
        ax_temp_annual.plot(ma_temp.index, ma_temp.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)
        
        # Plot annual moisture exceedances (transparent)
        ax_moisture_annual.plot(annual_moisture.index, annual_moisture.values, linewidth=1, marker='o', markersize=2, color=color, alpha=0.2)
        # Plot 20-year moving average on top
        ax_moisture_annual.plot(ma_moisture.index, ma_moisture.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)
    
    # Format cumulative temperature plot
    ax_temp_cum.set_ylabel('Cumulative Weeks', fontsize=12)
    ax_temp_cum.set_xlabel('Year', fontsize=12)
    ax_temp_cum.set_title(f'Cumulative Weeks Below Temperature Threshold - {species_label}\n(Temp < {Tmin}°C) - Malmö', 
                fontsize=12, fontweight='bold')
    ax_temp_cum.legend(title='Scenario', fontsize=10)
    ax_temp_cum.grid(True, alpha=0.3)
    
    # Format cumulative moisture plot
    ax_moisture_cum.set_ylabel('Cumulative Weeks', fontsize=12)
    ax_moisture_cum.set_xlabel('Year', fontsize=12)
    ax_moisture_cum.set_title(f'Cumulative Weeks Below Soil Moisture Threshold - {species_label}\n(W < {Rwmin}) - Malmö', 
                fontsize=12, fontweight='bold')
    ax_moisture_cum.legend(title='Scenario', fontsize=10)
    ax_moisture_cum.grid(True, alpha=0.3)
    
    # Format annual temperature plot
    ax_temp_annual.set_ylabel('Weeks per Year', fontsize=12)
    ax_temp_annual.set_xlabel('Year', fontsize=12)
    ax_temp_annual.set_title(f'Annual Weeks Below Temperature Threshold - {species_label}\n({ma_window_years}-Year Moving Average) - Malmö', 
                fontsize=12, fontweight='bold')
    ax_temp_annual.legend(title='Scenario', fontsize=10)
    ax_temp_annual.grid(True, alpha=0.3)
    
    # Format annual moisture plot
    ax_moisture_annual.set_ylabel('Weeks per Year', fontsize=12)
    ax_moisture_annual.set_xlabel('Year', fontsize=12)
    ax_moisture_annual.set_title(f'Annual Weeks Below Soil Moisture Threshold - {species_label}\n({ma_window_years}-Year Moving Average) - Malmö', 
                fontsize=12, fontweight='bold')
    ax_moisture_annual.legend(title='Scenario', fontsize=10)
    ax_moisture_annual.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'malmo_{species}_threshold_exceedances.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved species-specific threshold exceedances plot: {filename}")
    plt.show()

##########################################
### Create species-specific growth limitation plots

print("\n" + "="*70)
print("Creating species-specific growth limitation plots...")
print("="*70)

for species in SPECIES_TO_RUN:
    species_label = species.replace('_', ' ').title()
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
    
    for scenario in SCENARIOS_TO_RUN:
        # Load forcing data for this scenario
        path_forcing = your_folder + f"{scenario}.csv"
        Forcing = pd.read_csv(path_forcing)
        
        # Filter for Malmö
        Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
        
        # Calculate where both conditions are below threshold (limiting growth)
        both_below = ((Malmö_data['Temp'] < Tmin) | (Malmö_data['W'] < Rwmin)).astype(int)
        
        # Calculate cumulative counts
        cumulative_both = both_below.cumsum()
        
        # Calculate annual counts
        Malmö_data['both_below'] = both_below
        annual_both = Malmö_data.groupby('Year')['both_below'].sum()
        
        # Calculate 20-year moving average
        ma_both = annual_both.rolling(window=ma_window_years, center=True, min_periods=1).mean()
        
        # Create datetime for x-axis
        dates = pd.to_datetime(
            [f"{year}-W{week:02d}-1" for year, week in zip(Malmö_data['Year'].values, Malmö_data['Week'].values)],
            format='%Y-W%W-%w'
        )
        
        # Get color for this scenario
        color = scenario_colors.get(scenario, '#000000')
        
        # Plot cumulative combined threshold exceedances
        ax_left.plot(dates, cumulative_both, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.8)
        
        # Plot annual combined threshold exceedances (transparent)
        ax_right.plot(annual_both.index, annual_both.values, linewidth=1, marker='o', markersize=2, color=color, alpha=0.2)
        # Plot 20-year moving average on top
        ax_right.plot(ma_both.index, ma_both.values, linewidth=2.5, color=color, label=scenario.upper(), alpha=0.9)
    
    # Format left plot (cumulative)
    ax_left.set_ylabel('Cumulative Weeks', fontsize=12)
    ax_left.set_xlabel('Year', fontsize=12)
    ax_left.set_title(f'Cumulative Weeks with Growth Limitation - {species_label}\n(Temperature < {Tmin}°C OR Soil Moisture < {Rwmin}) - Malmö', 
                fontsize=12, fontweight='bold')
    ax_left.legend(title='Scenario', fontsize=10)
    ax_left.grid(True, alpha=0.3)
    
    # Format right plot (annual counts)
    ax_right.set_ylabel('Weeks per Year', fontsize=12)
    ax_right.set_xlabel('Year', fontsize=12)
    ax_right.set_title(f'Annual Weeks with Growth Limitation - {species_label}\n({ma_window_years}-Year Moving Average) - Malmö', 
                fontsize=12, fontweight='bold')
    ax_right.legend(title='Scenario', fontsize=10)
    ax_right.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'malmo_{species}_growth_limitation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved species-specific limitation plot: {filename}")
    plt.show()

##########################################
### Create growth rate plots for each species

print("\n" + "="*70)
print("Creating growth rate plots...")
print("="*70)

# Create a plot for each species
for species in SPECIES_TO_RUN:
    plt.figure(figsize=(12, 8))
    
    for scenario in SCENARIOS_TO_RUN:
        # Find data for this species in this scenario
        species_data = [d for d in growth_rate_data[scenario] if d['species'] == species]
        
        if species_data:
            data = species_data[0]
            years = np.array(data['years'])
            growth_rates = np.array(data['growth_rates'])
            residual_errors = np.array(data['residual_errors'])
            
            # Get color for this scenario
            color = scenario_colors.get(scenario, '#000000')
            
            # Plot error envelope (± residual standard error)
            plt.fill_between(years, 
                           growth_rates - residual_errors, 
                           growth_rates + residual_errors,
                           color=color,
                           alpha=0.2,
                           linewidth=0)
            
            # Plot growth rate line on top
            plt.plot(years, growth_rates, 
                    linewidth=1.5, 
                    color=color,
                    label=scenario.upper(),
                    marker='o',
                    markersize=2,
                    alpha=0.8)
    
    plt.xlabel('Starting Year', fontsize=12)
    plt.ylabel('Growth Rate (m/year)', fontsize=12)
    plt.title(f'Growth Rate vs Starting Year - {species.replace("_", " ").title()}\nMalmö (shaded area = ±1 Residual Std Error)', fontsize=14, fontweight='bold')
    plt.legend(title='Scenario', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = f'malmo_{species}_growth_rates_all_scenarios.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved growth rate plot: {filename}")
    plt.show()

# Create a combined plot with all species (only if more than one species)
if len(SPECIES_TO_RUN) > 1:
    fig, axes = plt.subplots(len(SPECIES_TO_RUN), 1, figsize=(14, 4 * len(SPECIES_TO_RUN)))
    
    # Handle single species case (axes is not an array)
    if len(SPECIES_TO_RUN) == 1:
        axes = [axes]
    
    for ax_idx, species in enumerate(SPECIES_TO_RUN):
        ax = axes[ax_idx]
        
        for scenario in SCENARIOS_TO_RUN:
            # Find data for this species in this scenario
            species_data = [d for d in growth_rate_data[scenario] if d['species'] == species]
            
            if species_data:
                data = species_data[0]
                years = np.array(data['years'])
                growth_rates = np.array(data['growth_rates'])
                residual_errors = np.array(data['residual_errors'])
                
                # Get color for this scenario
                color = scenario_colors.get(scenario, '#000000')
                
                # Plot error envelope (±1 residual standard error)
                ax.fill_between(years, 
                               growth_rates - residual_errors, 
                               growth_rates + residual_errors,
                               color=color,
                               alpha=0.2,
                               linewidth=0)
                
                # Plot growth rate line on top
                ax.plot(years, growth_rates, 
                       linewidth=1.5, 
                       color=color,
                       label=scenario.upper(),
                       marker='o',
                       markersize=2,
                       alpha=0.8)
        
        ax.set_xlabel('Starting Year', fontsize=11)
        ax.set_ylabel('Growth Rate (m/year)', fontsize=11)
        ax.set_title(f'{species.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(title='Scenario', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Growth Rate vs Starting Year - All Species (Malmö)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save combined figure
    filename = 'malmo_all_species_growth_rates_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined growth rate plot: {filename}")
    plt.show()

# Create species comparison plots for each scenario
print("\n" + "="*70)
print("Creating species comparison plots...")
print("="*70)

if len(SPECIES_TO_RUN) > 1:
    # Define colors for each species
    species_colors = {
        'abies_alba': '#1f77b4',      # Blue
        'picea_abies': '#ff7f0e',     # Orange
        'pseudo_menzii': '#2ca02c'    # Green
    }
    
    for scenario in SCENARIOS_TO_RUN:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        for species in SPECIES_TO_RUN:
            # Find data for this species in this scenario
            species_data = [d for d in growth_rate_data[scenario] if d['species'] == species]
            
            if species_data:
                data = species_data[0]
                years = np.array(data['years'])
                growth_rates = np.array(data['growth_rates'])
                residual_errors = np.array(data['residual_errors'])
                
                # Get color for this species
                color = species_colors.get(species, '#000000')
                species_label = species.replace('_', ' ').title()
                
                # Left plot: Raw growth rates with error envelope
                ax1.fill_between(years, 
                               growth_rates - residual_errors, 
                               growth_rates + residual_errors,
                               color=color,
                               alpha=0.15,
                               linewidth=0)
                ax1.plot(years, growth_rates, 
                        linewidth=1, 
                        color=color,
                        label=species_label,
                        marker='o',
                        markersize=1.5,
                        alpha=0.3)
                
                # Right plot: 20-year moving average
                # Create pandas Series for rolling window
                df = pd.DataFrame({'year': years, 'growth_rate': growth_rates})
                df = df.sort_values('year')
                points_per_year = 26  # ~26 data points per year (every 2 weeks)
                window_size = int(20 * points_per_year)
                df['ma'] = df['growth_rate'].rolling(window=window_size, center=True, min_periods=1).mean()
                
                ax2.plot(df['year'], df['ma'], 
                        linewidth=2.5, 
                        color=color,
                        label=species_label,
                        alpha=0.9)
        
        # Format left plot
        ax1.set_xlabel('Starting Year', fontsize=12)
        ax1.set_ylabel('Growth Rate (m/year)', fontsize=12)
        ax1.set_title(f'Growth Rate by Species - {scenario.upper()}\nMalmö (shaded = ±1 Residual Std Error)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(title='Species', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Format right plot
        ax2.set_xlabel('Starting Year', fontsize=12)
        ax2.set_ylabel('Growth Rate (m/year)', fontsize=12)
        ax2.set_title(f'Growth Rate by Species (20-Year Moving Average) - {scenario.upper()}\nMalmö', 
                     fontsize=12, fontweight='bold')
        ax2.legend(title='Species', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'malmo_species_comparison_{scenario}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved species comparison plot for {scenario.upper()}: {filename}")
        plt.show()

# Create overall species comparison across scenarios
if len(SPECIES_TO_RUN) > 1:
    print("\n" + "="*70)
    print("Creating overall species comparison across scenarios...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Define colors for species
    species_colors = {
        'abies_alba': '#1f77b4',
        'picea_abies': '#ff7f0e',
        'pseudo_menzii': '#2ca02c'
    }
    
    # Define line styles for scenarios
    scenario_linestyles = {
        'ssp126': '-',
        'ssp245': '--',
        'ssp370': '-.',
        'ssp585': ':'
    }
    
    for species in SPECIES_TO_RUN:
        species_label = species.replace('_', ' ').title()
        color = species_colors.get(species, '#000000')
        
        for scenario in SCENARIOS_TO_RUN:
            # Find data for this species in this scenario
            species_data = [d for d in growth_rate_data[scenario] if d['species'] == species]
            
            if species_data:
                data = species_data[0]
                years = np.array(data['years'])
                growth_rates = np.array(data['growth_rates'])
                
                # Create pandas Series for rolling window
                df = pd.DataFrame({'year': years, 'growth_rate': growth_rates})
                df = df.sort_values('year')
                points_per_year = 26
                window_size = int(20 * points_per_year)
                df['ma'] = df['growth_rate'].rolling(window=window_size, center=True, min_periods=1).mean()
                
                linestyle = scenario_linestyles.get(scenario, '-')
                label = f"{species_label} ({scenario.upper()})"
                
                # Plot on appropriate subplot based on scenario
                if scenario == 'ssp126':
                    axes[0, 0].plot(df['year'], df['ma'], linewidth=2, color=color, 
                                  linestyle=linestyle, label=species_label, alpha=0.9)
                elif scenario == 'ssp245':
                    axes[0, 1].plot(df['year'], df['ma'], linewidth=2, color=color, 
                                  linestyle=linestyle, label=species_label, alpha=0.9)
                elif scenario == 'ssp370':
                    axes[1, 0].plot(df['year'], df['ma'], linewidth=2, color=color, 
                                  linestyle=linestyle, label=species_label, alpha=0.9)
                else:  # ssp585
                    axes[1, 1].plot(df['year'], df['ma'], linewidth=2, color=color, 
                                  linestyle=linestyle, label=species_label, alpha=0.9)
    
    # Format subplots
    scenario_titles = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
    for idx, (ax, scenario_name) in enumerate(zip(axes.flatten(), scenario_titles)):
        ax.set_xlabel('Starting Year', fontsize=11)
        ax.set_ylabel('Growth Rate (m/year)', fontsize=11)
        ax.set_title(f'{scenario_name}', fontsize=12, fontweight='bold')
        ax.legend(title='Species', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Species Comparison: 20-Year Moving Average Growth Rates\nMalmö', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filename = 'malmo_species_comparison_all_scenarios.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved overall species comparison plot: {filename}")
    plt.show()

##########################################
### Create species comparison plots for threshold exceedances

print("\n" + "="*70)
print("Creating species comparison plots for threshold exceedances...")
print("="*70)

# Note: Threshold exceedances are climate-based and same for all species
# We create comparison plots showing all "species" together (though data is identical)
# to maintain consistency with other species comparison plots

if len(SPECIES_TO_RUN) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Define colors for species (for consistency)
    species_colors_threshold = {
        'abies_alba': '#1f77b4',
        'picea_abies': '#ff7f0e',
        'pseudo_menzii': '#2ca02c'
    }
    
    for scenario_idx, scenario in enumerate(SCENARIOS_TO_RUN):
        ax = axes[scenario_idx // 2, scenario_idx % 2]
        
        # Load forcing data for this scenario
        path_forcing = your_folder + f"{scenario}.csv"
        Forcing = pd.read_csv(path_forcing)
        
        # Filter for Malmö
        Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
        
        # Calculate where both conditions are below threshold (limiting growth)
        both_below = ((Malmö_data['Temp'] < Tmin) | (Malmö_data['W'] < Rwmin)).astype(int)
        
        # Calculate annual counts
        Malmö_data['both_below'] = both_below
        annual_both = Malmö_data.groupby('Year')['both_below'].sum()
        
        # Calculate 20-year moving average
        ma_both = annual_both.rolling(window=ma_window_years, center=True, min_periods=1).mean()
        
        # Since threshold data is the same for all species (climate-based),
        # we'll plot it once per scenario
        color = scenario_colors.get(scenario, '#000000')
        
        # Plot annual combined threshold exceedances (transparent)
        ax.plot(annual_both.index, annual_both.values, linewidth=1, marker='o', 
               markersize=2, color=color, alpha=0.2, label='Annual Count')
        # Plot 20-year moving average on top
        ax.plot(ma_both.index, ma_both.values, linewidth=2.5, color=color, 
               label=f'{ma_window_years}-Year MA', alpha=0.9)
        
        # Format plot
        ax.set_ylabel('Weeks per Year', fontsize=11)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_title(f'{scenario.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Growth Limitation Across Scenarios\n(Temperature < {Tmin}°C OR Soil Moisture < {Rwmin}) - Malmö', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filename = 'malmo_threshold_exceedances_comparison_all_scenarios.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved threshold exceedances comparison plot: {filename}")
    plt.show()

##########################################
### Create species comparison plots for growth limitation (by scenario)

print("\n" + "="*70)
print("Creating species comparison plots for growth limitation...")
print("="*70)

if len(SPECIES_TO_RUN) > 1:
    for scenario in SCENARIOS_TO_RUN:
        fig, (ax_cumulative, ax_annual) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Load forcing data for this scenario
        path_forcing = your_folder + f"{scenario}.csv"
        Forcing = pd.read_csv(path_forcing)
        
        # Filter for Malmö
        Malmö_data = Forcing[Forcing['Location'] == "Malmö"].copy()
        
        # Calculate where both conditions are below threshold (limiting growth)
        both_below = ((Malmö_data['Temp'] < Tmin) | (Malmö_data['W'] < Rwmin)).astype(int)
        
        # Calculate cumulative counts
        cumulative_both = both_below.cumsum()
        
        # Calculate annual counts
        Malmö_data['both_below'] = both_below
        annual_both = Malmö_data.groupby('Year')['both_below'].sum()
        
        # Calculate 20-year moving average
        ma_both = annual_both.rolling(window=ma_window_years, center=True, min_periods=1).mean()
        
        # Create datetime for x-axis
        dates = pd.to_datetime(
            [f"{year}-W{week:02d}-1" for year, week in zip(Malmö_data['Year'].values, Malmö_data['Week'].values)],
            format='%Y-W%W-%w'
        )
        
        # Get color for this scenario
        color = scenario_colors.get(scenario, '#000000')
        
        # Plot cumulative combined threshold exceedances
        ax_cumulative.plot(dates, cumulative_both, linewidth=2.5, color=color, 
                          label='All Species (Climate-based)', alpha=0.8)
        
        # Plot annual combined threshold exceedances (transparent)
        ax_annual.plot(annual_both.index, annual_both.values, linewidth=1, marker='o', 
                      markersize=2, color=color, alpha=0.2)
        # Plot 20-year moving average on top
        ax_annual.plot(ma_both.index, ma_both.values, linewidth=2.5, color=color, 
                      label='All Species (Climate-based)', alpha=0.9)
        
        # Format cumulative plot
        ax_cumulative.set_ylabel('Cumulative Weeks', fontsize=12)
        ax_cumulative.set_xlabel('Year', fontsize=12)
        ax_cumulative.set_title(f'Cumulative Weeks with Growth Limitation - {scenario.upper()}\n(Temp < {Tmin}°C OR W < {Rwmin}) - Malmö', 
                    fontsize=12, fontweight='bold')
        ax_cumulative.legend(fontsize=10)
        ax_cumulative.grid(True, alpha=0.3)
        
        # Format annual plot
        ax_annual.set_ylabel('Weeks per Year', fontsize=12)
        ax_annual.set_xlabel('Year', fontsize=12)
        ax_annual.set_title(f'Annual Weeks with Growth Limitation - {scenario.upper()}\n({ma_window_years}-Year Moving Average) - Malmö', 
                    fontsize=12, fontweight='bold')
        ax_annual.legend(fontsize=10)
        ax_annual.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'malmo_growth_limitation_comparison_{scenario}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved growth limitation comparison plot for {scenario.upper()}: {filename}")
        plt.show()

# Create moving average plots for growth rates
print("\n" + "="*70)
print("Creating moving average plots for growth rates...")
print("="*70)

# Define window size for moving average (in years)
window_years = 20

for species in SPECIES_TO_RUN:
    plt.figure(figsize=(12, 8))
    
    for scenario in SCENARIOS_TO_RUN:
        # Find data for this species in this scenario
        species_data = [d for d in growth_rate_data[scenario] if d['species'] == species]
        
        if species_data:
            data = species_data[0]
            years = np.array(data['years'])
            growth_rates = np.array(data['growth_rates'])
            
            # Create pandas Series for easy rolling window calculation
            df = pd.DataFrame({'year': years, 'growth_rate': growth_rates})
            df = df.sort_values('year')
            
            # Calculate number of data points per year (approximately)
            # Since we have data every 2 weeks, that's roughly 26 data points per year
            points_per_year = 26
            window_size = int(window_years * points_per_year)
            
            # Calculate moving average and moving standard deviation
            df['ma'] = df['growth_rate'].rolling(window=window_size, center=True, min_periods=1).mean()
            df['std'] = df['growth_rate'].rolling(window=window_size, center=True, min_periods=1).std()
            
            # Get color for this scenario
            color = scenario_colors.get(scenario, '#000000')
            
            # Plot the envelope (mean ± std)
            plt.fill_between(df['year'], 
                           df['ma'] - df['std'], 
                           df['ma'] + df['std'],
                           color=color,
                           alpha=0.2,
                           linewidth=0)
            
            # Plot the moving average line on top
            plt.plot(df['year'], df['ma'], 
                    linewidth=2.5, 
                    color=color,
                    label=scenario.upper(),
                    alpha=0.9)
    
    plt.xlabel('Starting Year', fontsize=12)
    plt.ylabel(f'Growth Rate - {window_years}-Year Moving Average (m/year)', fontsize=12)
    plt.title(f'Growth Rate Moving Average ({window_years}-Year Window) - {species.replace("_", " ").title()}\nMalmö (shaded area = ±1 std dev)', 
             fontsize=14, fontweight='bold')
    plt.legend(title='Scenario', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = f'malmo_{species}_growth_rates_ma{window_years}yr_all_scenarios.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved moving average plot ({window_years}-year): {filename}")
    plt.show()

print("\n" + "="*70)
print("All plots created successfully!")
print("="*70)
