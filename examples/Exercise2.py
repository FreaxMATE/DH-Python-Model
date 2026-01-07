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
path_forcing = your_folder + "ssp126.csv"
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
# plt.plot(Forcing[H_idx]['Temp'].values, label='Härnösand', color='red')
# plt.plot(Forcing[L_idx]['Temp'].values, label='Luleå', color='green')
# plt.legend()
# plt.ylabel('Temperature (°C)')
# plt.ylim(-50, 30)
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
## 6 years, 53 weeks:

idx_bl = list(range(0, 6*53))  # Select indices to get 6 years (6*53 weeks) of forcing

##########################################
### Malmö

print("\n" + "="*50)
print("Running model for Malmö...")
print("="*50)

Malmö = run_dh_model(
    Tair=Malmö_clim['Temp'].values[idx_bl],
    Rw=Malmö_clim['W'].values[idx_bl],
    Rw_vol=False,
    Cpool=Malmö_clim['Cpool'].values[idx_bl],
    params=DH_model_ins['Parameters']['best'],
    week=Malmö_clim['Week'].values[idx_bl],
    year=Malmö_clim['Year'].values[idx_bl],
    DH_plot=True
)

# Plot cumulative increment
plt.figure(figsize=(10, 6))
cumulative_incr = np.cumsum(Malmö['Incr'].values * 2)  # *2 from radius to diameter
colors = (Malmö['years'].values - 2014)
scatter = plt.scatter(range(len(cumulative_incr)), cumulative_incr, 
                     c=colors, s=10, cmap='viridis')
plt.colorbar(scatter, label='Years since 2014')
plt.ylabel('Cumulative increment (mm)')
plt.xlabel('Week')
plt.title('Malmö - Baseline Growth')
plt.grid(True, alpha=0.3)
plt.show()

##########################################################################################
# OK, great,.. now we have a model that maybe runs for the baseline years
# we want it to run. It outputs diameter, .. what now?

# Skim through publication:
# Pretzsch, H., Biber, P., Uhl, E., Dahlhausen, J., Rötzer, T., Caldentey, J.,
# Koike, T., Van Con, T., Chavanne, A., Seifert, T., Toit, B. D., Farnden, C.,
# & Pauleit, S. (2015). Crown size and growing space requirement of common
# tree species in urban centres, parks, and forests. Urban Forestry & Urban Greening,
# 14(3), 466–479. https://doi.org/10.1016/j.ufug.2015.04.006

# Find the equation and species relevant to "transforming" diameter to height.

##########################################################################################
### Malmö
# Using the equation and the parameters you found, transform your Incr (cm) to height:
# Parameters for a species sensible as Christmas tree.

# Put the code and parameters you work out from the publication here.
# Example (adjust parameters based on the publication):
# from dhmodel.utils import calculate_height_from_diameter
# diameter_cm = np.cumsum(Malmö['Incr'].values * 2) / 10  # Convert mm to cm
# height_m = calculate_height_from_diameter(diameter_cm, a=1.2, b=0.6)
# plt.plot(height_m)
# plt.ylabel('Height (m)')
# plt.xlabel('Week')
# plt.title('Malmö - Tree Height')
# plt.show()

#### Tree grows too much during baseline period:
# What parameters would be useful/sensible(!) to change, in order to obtain
# roughly 1.5 meter during the baseline period?

# params_update = DH_model_ins['Parameters']['best'].copy()
# params_update[0] = 6.0  # Example: adjust Tmin
# # Or adjust other parameters like D_n_max, b, etc.

# Rerun with updated parameters:
# Malmö = run_dh_model(
#     Tair=Malmö_clim['Temp'].values[idx_bl],
#     Rw=Malmö_clim['W'].values[idx_bl],
#     Rw_vol=False,
#     Cpool=Malmö_clim['Cpool'].values[idx_bl],
#     params=params_update,
#     week=Malmö_clim['Week'].values[idx_bl],
#     year=Malmö_clim['Year'].values[idx_bl],
#     DH_plot=True
# )
#
# plt.plot(np.cumsum(Malmö['Incr'].values * 2), color=(Malmö['years'].values - 2014))
# plt.show()

#############################################
#### If the Tree still grows too much/little during baseline period.
# Change parameter more or turn to a different parameter.
# What else can be changed sensibly?

##########################################################################################
### Luleå

print("\n" + "="*50)
print("Running model for Luleå...")
print("="*50)

# Using the equation and the parameters you found, transform your Incr (cm) to height:
# Parameters for a species sensible as Christmas tree.

# Put the code and parameters you work out from the publication here.

#### Tree grows too much or little during baseline period:
# What parameters would be useful/sensible(!) to change, in order to obtain
# roughly 1.5 meter during the baseline period?

# params_update = DH_model_ins['Parameters']['best'].copy()
# # Put the updated parameter set here

# Rerun with updated parameters:
Luleå = run_dh_model(
    Tair=Luleå_clim['Temp'].values[idx_bl],
    Rw=Luleå_clim['W'].values[idx_bl],
    Rw_vol=False,
    Cpool=Luleå_clim['Cpool'].values[idx_bl],
    params=DH_model_ins['Parameters']['best'],  # Use params_update when ready
    week=Luleå_clim['Week'].values[idx_bl],
    year=Luleå_clim['Year'].values[idx_bl],
    DH_plot=True
)

# Add to your reference parameter graph to see the difference:
# plt.plot(np.cumsum(Luleå['Incr'].values * 2), color=(Luleå['years'].values - 2014))

##########################################################################################
### Härnösand

print("\n" + "="*50)
print("Running model for Härnösand...")
print("="*50)

# Using the equation and the parameters you found, transform your Incr (cm) to height:
# Parameters for a species sensible as Christmas tree.

# Put the code and parameters you work out from the publication here.

#### Tree grows too much or little during baseline period:
# What parameters would be useful/sensible(!) to change, in order to obtain
# roughly 1.5 meter during the baseline period?

# params_update = DH_model_ins['Parameters']['best'].copy()
# Put the updated parameter set here

# Rerun with updated parameters:
Härnösand = run_dh_model(
    Tair=Härnösand_clim['Temp'].values[idx_bl],
    Rw=Härnösand_clim['W'].values[idx_bl],
    Rw_vol=False,
    Cpool=Härnösand_clim['Cpool'].values[idx_bl],
    params=DH_model_ins['Parameters']['best'],  # Use params_update when ready
    week=Härnösand_clim['Week'].values[idx_bl],
    year=Härnösand_clim['Year'].values[idx_bl],
    DH_plot=True
)

# plt.plot(np.cumsum(Härnösand['Incr'].values * 2), color=(Härnösand['years'].values - 2014))

##########################################################################################
# Some quick questions you could answer -- they may not be relevant to the impact
# assessment, but they get you an understanding of what may be possible with the data

# What does this baseline parameter set lead to at the end of the century
# for the different ssps?

# idx_end = list(range(len(Härnösand_clim) - 6*53, len(Härnösand_clim)))
# These are the last years in the timeseries

# Härnösand_end = run_dh_model(
#     Tair=Härnösand_clim['Temp'].values[idx_end],
#     Rw=Härnösand_clim['W'].values[idx_end],
#     Rw_vol=False,
#     Cpool=Härnösand_clim['Cpool'].values[idx_end],
#     params=params_update,
#     week=Härnösand_clim['Week'].values[idx_end],
#     year=Härnösand_clim['Year'].values[idx_end],
#     DH_plot=False
# )
#
# plt.plot(np.cumsum(Härnösand_end['Incr'].values * 2),
#          color=(Härnösand_end['years'].values - 2014))

# Ok, now that the baseline trees are set up, you can start with your project..

print("\n" + "="*50)
print("Baseline runs complete!")
print("="*50)
print("\nNext steps:")
print("1. Calibrate parameters to achieve ~1.5m height in baseline period")
print("2. Run scenarios for different SSPs")
print("3. Analyze climate change impacts on tree growth")
print("4. Compare growth across locations")

############################################################################################################
# Example: Graph used in lecture (comparison of SSPs)
# Uncomment to use:

# from dhmodel.utils import aggregate_by_year
# 
# # Load different scenarios
# scenarios = {}
# for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
#     path = your_folder + f"{ssp}.csv"
#     scenarios[ssp] = pd.read_csv(path)
# 
# # Plot for each location
# fig, axes = plt.subplots(3, 1, figsize=(12, 12))
# colors = {'ssp126': 'green', 'ssp245': 'orange', 'ssp370': 'red', 'ssp585': 'purple'}
# 
# for idx, site in enumerate(['Luleå', 'Härnösand', 'Malmö']):
#     ax = axes[idx]
#     for ssp_name, ssp_data in scenarios.items():
#         agg = aggregate_by_year(ssp_data, site, 'Temp')
#         ax.plot(agg['Year'], agg['Temp'], color=colors[ssp_name], 
#                 label=ssp_name, linewidth=2)
#     ax.set_ylabel('Temperature (°C)')
#     ax.set_title(site)
#     ax.set_ylim(0, 15)
#     ax.grid(True, alpha=0.3)
#     if idx == 2:
#         ax.set_xlabel('Year')
#         ax.legend(loc='upper left')
# 
# plt.tight_layout()
# plt.show()
