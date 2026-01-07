"""
Data loading utilities for DH-model.

Provides example forcing data and parameters for running the model.
"""

import pickle
import os
import pandas as pd
import numpy as np


def load_dh_model_ins():
    """
    Load example forcing data and parameters needed for running the DH model.
    
    Contains lists with data frames containing forcing data for two sites in France 
    (GRA and ABR), alongside parameters with best, upper and lower limits.
    
    Data obtained from Cyrille Rathgeber and used in:
    http://www.plantphysiol.org/lookup/doi/10.1104/pp.16.00037
    
    Modified to weekly resolution. Cpool was obtained from Ogee et al Figure 3, 
    and scaled for the purposes here. These are site data from locations where 
    xylogenesis and wood anatomical data exists.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'Inputs_GRA_3years': DataFrame with forcing data for GRA site (3 years)
        - 'Inputs_ABR_4years': DataFrame with forcing data for ABR site (4 years)
        - 'Parameters': Dictionary with 'best', 'upper', and 'lower' parameter values
    
    Examples
    --------
    >>> from dhmodel import load_dh_model_ins
    >>> data = load_dh_model_ins()
    >>> print(data.keys())
    dict_keys(['Inputs_GRA_3years', 'Inputs_ABR_4years', 'Parameters'])
    >>> 
    >>> # Access parameters
    >>> params_best = data['Parameters']['best']
    >>> 
    >>> # Access forcing data
    >>> forcing_gra = data['Inputs_GRA_3years']
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DH_model_ins.pkl')
    
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Return example data structure if file doesn't exist
        print("Warning: DH_model_ins.pkl not found. Creating example data structure.")
        return _create_example_data()


def _create_example_data():
    """
    Create example data structure for demonstration.
    
    Creates synthetic forcing data with realistic seasonal patterns.
    Parameters are from Wilkinson et al 2015 and Cuny & Rathgeber 2016.
    """
    # Example parameters from Wilkinson et al 2015
    # [Tmin, Rwmin, D_n_max, b, D_v_max, delta, chi, R_w_crit, D_m_max, dayl_min]
    params_best = np.array([5.0, 0.15, 0.8, 0.15, 0.0052, 0.5, 4.0, 0.4, 0.0045, 12.0])
    params_upper = np.array([7.0, 0.20, 1.0, 0.20, 0.0065, 0.7, 5.0, 0.5, 0.0055, 14.0])
    params_lower = np.array([3.0, 0.10, 0.6, 0.10, 0.0040, 0.3, 3.0, 0.3, 0.0035, 10.0])
    
    # Create example forcing data for GRA site (3 years, 53 weeks/year)
    n_weeks_gra = 3 * 53
    weeks_gra = np.tile(np.arange(1, 54), 3)[:n_weeks_gra]
    years_gra = np.repeat([2013, 2014, 2015], 53)[:n_weeks_gra]
    
    # Simulate seasonal temperature pattern (realistic for temperate climate)
    t_gra = np.linspace(0, 3 * 2 * np.pi, n_weeks_gra)
    Tair_gra = 12 + 8 * np.sin(t_gra - np.pi/2)
    
    # Simulate soil moisture with some inter-annual variability
    SW_gra = 0.6 + 0.15 * np.sin(t_gra + np.pi/4) + 0.05 * np.sin(t_gra * 3)
    SW_gra = np.clip(SW_gra, 0.2, 0.9)
    
    # Simulate carbon pool with seasonal pattern
    Cpool_gra = 2.0 + 0.8 * np.sin(t_gra + np.pi/3) + 0.3 * np.sin(t_gra * 2)
    Cpool_gra = np.clip(Cpool_gra, 0.5, 3.5)
    
    inputs_gra = pd.DataFrame({
        'Tair': Tair_gra,
        'SW': SW_gra,
        'Cpool': Cpool_gra,
        'week': weeks_gra,
        'year.datetime.': years_gra
    })
    
    # Create example forcing data for ABR site (4 years)
    n_weeks_abr = 4 * 53
    weeks_abr = np.tile(np.arange(1, 54), 4)[:n_weeks_abr]
    years_abr = np.repeat([2013, 2014, 2015, 2016], 53)[:n_weeks_abr]
    
    # Slightly different climate for ABR site
    t_abr = np.linspace(0, 4 * 2 * np.pi, n_weeks_abr)
    Tair_abr = 11 + 9 * np.sin(t_abr - np.pi/2)
    
    SW_abr = 0.55 + 0.2 * np.sin(t_abr + np.pi/4) + 0.05 * np.sin(t_abr * 3)
    SW_abr = np.clip(SW_abr, 0.15, 0.95)
    
    Cpool_abr = 1.8 + 0.9 * np.sin(t_abr + np.pi/3) + 0.3 * np.sin(t_abr * 2)
    Cpool_abr = np.clip(Cpool_abr, 0.4, 3.2)
    
    inputs_abr = pd.DataFrame({
        'Tair': Tair_abr,
        'SW': SW_abr,
        'Cpool': Cpool_abr,
        'week': weeks_abr,
        'year.datetime.': years_abr
    })
    
    return {
        'Inputs_GRA_3years': inputs_gra,
        'Inputs_ABR_4years': inputs_abr,
        'Parameters': {
            'best': params_best,
            'upper': params_upper,
            'lower': params_lower
        }
    }
