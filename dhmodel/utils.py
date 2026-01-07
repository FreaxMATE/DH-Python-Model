"""
Utility functions for data processing and analysis.
"""

import pandas as pd
import numpy as np


def aggregate_by_year(df, location, variable='Temp'):
    """
    Aggregate climate data by year.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with climate data containing 'Location', 'Year', and variable columns
    location : str
        Location name to filter
    variable : str, optional
        Variable name to aggregate (default: 'Temp')
    
    Returns
    -------
    pd.DataFrame
        Aggregated data by year with mean values
    
    Examples
    --------
    >>> import pandas as pd
    >>> from dhmodel.utils import aggregate_by_year
    >>> 
    >>> # Assuming you have loaded forcing data
    >>> forcing = pd.read_csv('ssp126.csv')
    >>> malmo_annual = aggregate_by_year(forcing, 'Malmö', 'Temp')
    """
    filtered = df[df['Location'] == location].copy()
    return filtered.groupby('Year').agg({variable: 'mean'}).reset_index()


def calculate_height_from_diameter(diameter, a=-0.1753, alpha=0.9113):
    """
    Transform diameter to height using logarithmic allometric equation.
    
    Formula: h = exp(a + alpha * log(diameter_cm))
    where diameter is converted from mm to cm by dividing by 10.
    
    Parameters
    ----------
    diameter : float or array-like
        Diameter values in cm
    a : float, optional
        Intercept parameter (default: -0.1753)
    alpha : float, optional
        Slope parameter (default: 0.9113)
    
    Returns
    -------
    float or array-like
        Height values in meters
    
    Examples
    --------
    >>> from dhmodel.utils import calculate_height_from_diameter
    >>> 
    >>> # For Norway spruce with default parameters
    >>> diameter_cm = 5.0  # cm
    >>> height_m = calculate_height_from_diameter(diameter_cm)
    >>> print(f"Height: {height_m:.2f} m")
    >>> 
    >>> # Using custom parameters
    >>> height_m = calculate_height_from_diameter(diameter_cm, a=-0.2, alpha=0.9)
    >>> print(f"Height: {height_m:.2f} m")
    """
    diameter = np.asarray(diameter)
    h_log = a + alpha * np.log(diameter)
    height = np.exp(h_log)
    return height


def convert_increment_to_diameter(incr_mm, n_years=1):
    """
    Convert radial increment to diameter increment.
    
    Parameters
    ----------
    incr_mm : array-like
        Radial increment in mm
    n_years : int, optional
        Number of years to cumulate (default: 1)
    
    Returns
    -------
    array-like
        Diameter increment in mm (radius * 2)
    
    Examples
    --------
    >>> import numpy as np
    >>> from dhmodel.utils import convert_increment_to_diameter
    >>> 
    >>> # Weekly increments for 6 years
    >>> weekly_incr = np.random.uniform(0, 0.5, 6*53)
    >>> diameter_mm = convert_increment_to_diameter(weekly_incr)
    """
    incr_mm = np.asarray(incr_mm)
    return np.cumsum(incr_mm) * 2  # Convert radius to diameter


def plot_climate_scenarios(forcing_data_dict, location, variable='Temp'):
    """
    Plot multiple climate scenarios for comparison.
    
    Parameters
    ----------
    forcing_data_dict : dict
        Dictionary with scenario names as keys and DataFrames as values
    location : str
        Location to plot
    variable : str, optional
        Variable to plot (default: 'Temp')
    
    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from dhmodel.utils import plot_climate_scenarios
    >>> 
    >>> # Load multiple scenarios
    >>> scenarios = {
    ...     'ssp126': pd.read_csv('ssp126.csv'),
    ...     'ssp370': pd.read_csv('ssp370.csv'),
    ...     'ssp585': pd.read_csv('ssp585.csv')
    ... }
    >>> 
    >>> plot_climate_scenarios(scenarios, 'Malmö', 'Temp')
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    colors = {'ssp126': 'green', 'ssp245': 'orange', 
              'ssp370': 'red', 'ssp585': 'purple'}
    
    plt.figure(figsize=(12, 6))
    
    for scenario_name, df in forcing_data_dict.items():
        df_agg = aggregate_by_year(df, location, variable)
        color = colors.get(scenario_name, 'black')
        plt.plot(df_agg['Year'], df_agg[variable], 
                label=scenario_name, color=color, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel(f'{variable}')
    plt.title(f'{variable} projections for {location}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()
