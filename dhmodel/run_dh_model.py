"""
DH-model main function implementation.

Based on Deleuze et Houllier 1998 with modifications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


def run_dh_model(Tair, Rw, Cpool, params, week, year, 
                 Rw_vol=False, dayl=None, DH_plot=True):
    """
    Run the DH-model based on Deleuze et Houllier 1998.
    
    Temperature is translated into number of cells produced,
    Relative water content is translated into volume gain,
    Cpool is translated into mass gain. The model calculates the resulting density.
    
    Parameters
    ----------
    Tair : array-like
        Vector with weekly average air temperature (°C)
    Rw : array-like
        Vector with weekly average relative soil water content
    Cpool : array-like
        Vector with weekly average nonstructural carbohydrates (Kg[C])
    params : array-like
        Vector of parameter values [Tmin, Rwmin, D_n_max, b, D_v_max, 
                                    delta, chi, R_w_crit, D_m_max, dayl_min]
    week : array-like
        Vector with week number of a given year
    year : array-like
        Vector with repeated year entries for a given year
    Rw_vol : bool, optional
        Whether input for soil moisture is given as volumetric (True) or 
        relative soil (False) water content. Default: False
    dayl : array-like, optional
        Vector with weekly average daylength (hours). Can be created using 
        the daylength function. Daylength for growth initialisation is an 
        addition to the DH-model. Default: None
    DH_plot : bool, optional
        Whether output should be plotted or not. Default: True
    
    Returns
    -------
    pd.DataFrame
        Data frame with weekly entries of simulated density (Dens), 
        ring increment (Incr in mm), cell numbers (Nr), weeks, and years
    
    Examples
    --------
    >>> from dhmodel import load_dh_model_ins, run_dh_model
    >>> 
    >>> # Load example data and parameters
    >>> data = load_dh_model_ins()
    >>> 
    >>> # Run model
    >>> DH_out = run_dh_model(
    ...     Tair=data['Inputs_ABR_4years']['Tair'],
    ...     Rw=data['Inputs_ABR_4years']['SW'],
    ...     Rw_vol=False,
    ...     Cpool=data['Inputs_ABR_4years']['Cpool'],
    ...     params=data['Parameters']['best'],
    ...     week=data['Inputs_ABR_4years']['week'],
    ...     year=data['Inputs_ABR_4years']['year.datetime.'],
    ...     DH_plot=True
    ... )
    """
    # Convert inputs to numpy arrays
    Tair = np.asarray(Tair)
    Rw = np.asarray(Rw)
    Cpool = np.asarray(Cpool)
    week = np.asarray(week)
    year = np.asarray(year)
    
    w = len(week)
    D_n = np.zeros(w)
    D_v = np.zeros(w)
    D_m = np.zeros(w)
    
    # Extract parameters
    Tmin = params[0]       # °C Temperature below which cambial activity does not occur
    Rwmin = params[1]      # (-) minimum relative soil moisture
    D_n_max = params[2]    # Maximum cell production rate
    b = params[3]          # Temperature response parameter
    D_v_max = params[4]    # Maximum volume increment
    delta = params[5]      # Carbon response parameter
    chi = params[6]        # Soil moisture adjustment parameter
    R_w_crit = params[7]   # Critical soil moisture
    D_m_max = params[8]    # Maximum mass increment
    dayl_min = params[9] if len(params) > 9 else 0  # Minimum daylength
    
    # Convert volumetric to relative soil moisture if needed
    # Function from Wilkinson et al 2015
    if Rw_vol:
        Rw = 1 - 1 / (1 + (Rw / R_w_crit) ** chi)
    
    ds = np.arange(1, w + 1)
    
    # Run DH-model
    if dayl is not None:
        dayl = np.asarray(dayl)
        # If daylength vector is provided
        for i in range(w):
            if Tair[i] >= Tmin and Rw[i] >= Rwmin and dayl[i] >= dayl_min:
                D_n[i] = D_n_max * (1 - np.exp(-b * (Tair[i] - Tmin)))
                D_v[i] = D_v_max * Rw[i]
                D_m[i] = D_m_max * (1 - np.exp(-delta * Cpool[i]))
    else:
        # Without daylength constraint
        for i in range(w):
            if Tair[i] >= Tmin and Rw[i] >= Rwmin:
                D_n[i] = D_n_max * (1 - np.exp(-b * (Tair[i] - Tmin)))
                D_v[i] = D_v_max * Rw[i]
                D_m[i] = D_m_max * (1 - np.exp(-delta * Cpool[i]))
    
    # Prepare output and plotting
    Tair_df = pd.DataFrame({'Tair': Tair, 'dates': ds, 'D_n': D_n})
    Rw_df = pd.DataFrame({'Rw': Rw, 'dates': ds, 'D_v': D_v})
    Cpool_df = pd.DataFrame({'Cpool': Cpool, 'dates': ds, 'D_m': D_m})
    
    # Calculate increment and density
    # Get ring increment in mm, making some assumptions:
    # convert m3 to mm3 / (average cell tangential width = 20 micrometer, converted to mm) 
    # / (length of average cell = 1 meter, converted to mm)
    Incr = D_n * D_v
    Incr = Incr * 1e9 / (20 * 1000) / (1 * 1000)
    
    # Calculate density (g/cm³)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Dens = D_m / D_v
    
    Dens_df = pd.DataFrame({'Dens': Dens, 'weeks': week, 'years': year})
    
    # Plotting
    if DH_plot:
        _plot_dh_results(Tair_df, Rw_df, Cpool_df, Dens_df, Tmin, Rwmin, ds)
    
    # Create output dataframe
    df = Dens_df.copy()
    df['Incr'] = Incr
    df['Nr'] = D_n
    
    # Post-processing of NaNs to NA (more sensible for post-processing and plotting)
    df['Dens'] = df['Dens'].replace([np.inf, -np.inf], np.nan)
    df.loc[df['Dens'].isna(), 'Dens'] = np.nan
    
    return df


def _plot_dh_results(Tair_df, Rw_df, Cpool_df, Dens_df, Tmin, Rwmin, ds):
    """
    Helper function to create DH model plots.
    
    Creates a 4-panel plot showing:
    1. Temperature and cell production
    2. Soil moisture and volume increment
    3. Carbon pool and mass increment
    4. Wood density
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.05)
    
    # Temperature plot
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(Tair_df['dates'], Tair_df['Tair'], 'k-')
    ax1.set_ylim(0, 30)
    ax1.set_ylabel('Temperature (°C)', color='black')
    ax1.axhline(y=25, color='lightgray', linestyle='--', linewidth=0.5)
    ax1.axhline(y=Tmin, color='lightgray', linestyle='--', linewidth=0.5)
    ax1.set_xticklabels([])
    ax1.set_xlim(ds[0], ds[-1])
    ax1_twin.plot(Tair_df['dates'], Tair_df['D_n'], 'r.', markersize=2)
    ax1_twin.set_ylim(0, 1.5)
    ax1_twin.set_ylabel('cells/week', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Soil moisture plot
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(Rw_df['dates'], Rw_df['Rw'], 'k-')
    ax2.set_ylim(0, 1.52)
    ax2.set_ylabel('Rw (-)', color='black')
    ax2.axhline(y=Rwmin, color='lightgray', linestyle='--', linewidth=0.5)
    ax2.set_xticklabels([])
    ax2.set_xlim(ds[0], ds[-1])
    ax2_twin.plot(Rw_df['dates'], Rw_df['D_v'], 'r.', markersize=2)
    ax2_twin.set_ylim(0, 0.0074)
    ax2_twin.set_ylabel('Δ volume m³/week', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # Carbon pool plot
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    ax3.plot(Cpool_df['dates'], Cpool_df['Cpool'], 'k-')
    ax3.set_ylim(0, 4)
    ax3.set_ylabel('Cpool (Kg[C])', color='black')
    ax3.set_xticklabels([])
    ax3.set_xlim(ds[0], ds[-1])
    ax3_twin.plot(Cpool_df['dates'], Cpool_df['D_m'], 'r.', markersize=2)
    ax3_twin.set_ylim(0, 0.0052)
    ax3_twin.set_ylabel('Δ mass kg/week', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Density plot
    ax4 = axes[3]
    ax4.plot(Cpool_df['dates'], Dens_df['Dens'], 'k-')
    ax4.set_ylim(0, 2)
    ax4.set_ylabel('Density (g/cm³)', color='black')
    ax4.set_xlim(ds[0], ds[-1])
    
    # Add year labels to x-axis
    year_starts = np.where(Dens_df['weeks'].values == 1)[0]
    if len(year_starts) > 0:
        year_labels = Dens_df['years'].values[year_starts]
        ax4.set_xticks(year_starts + 1)  # +1 because dates start at 1
        ax4.set_xticklabels(year_labels.astype(int))
    ax4.set_xlabel('Years')
    
    plt.tight_layout()
    plt.show()
