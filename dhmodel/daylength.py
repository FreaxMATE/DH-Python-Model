"""
Daylength calculation function.

Based on the BASFOR model by David Cameron and Marcel van Oijen.
"""

import numpy as np


def daylength(sangle, LAT):
    """
    Calculate daylength based on solar angle and latitude.
    
    This function calculates daylength based on solar angle and latitude as input. 
    The Daylength Function is based on the BASFOR model by David Cameron and Marcel van Oijen.
    
    Parameters
    ----------
    sangle : float or array-like
        Solar angle in radians
    LAT : float
        Latitude at which the model is run (in degrees)
    
    Returns
    -------
    float or array-like
        Daylength in hours
    
    Examples
    --------
    >>> import numpy as np
    >>> from dhmodel import daylength
    >>> 
    >>> # Get solar angle for a year
    >>> day = np.arange(1, 365 * 4 + 1)
    >>> sangle = -23.45 * np.pi/180 * np.cos((np.pi/180) * (day + 10))
    >>> 
    >>> # Calculate daylength
    >>> dayl = np.array([daylength(s, LAT=48.6542) for s in sangle])
    >>> 
    >>> # Or vectorized:
    >>> dayl = daylength(sangle, LAT=48.6542)
    """
    RAD = np.pi / 180.0  # radians per degree
    
    # Convert inputs to numpy arrays for vectorization
    sangle = np.asarray(sangle)
    is_scalar = sangle.ndim == 0
    if is_scalar:
        sangle = sangle[np.newaxis]
    
    # Calculate DECC (declination corrected)
    # Amplitude of ~(-40, 40)
    lower_bound = np.arctan(-1.0 / np.tan(RAD * LAT))
    upper_bound = np.arctan(1.0 / np.tan(RAD * LAT))
    DECC = np.maximum(lower_bound, np.minimum(upper_bound, sangle))
    
    # Calculate daylength (fraction of day)
    DAYL = 0.5 * (1.0 + 2.0 * np.arcsin(np.tan(RAD * LAT) * np.tan(DECC)) / np.pi)
    
    # Convert to hours
    DAYL = DAYL * 24
    
    if is_scalar:
        return float(DAYL[0])
    return DAYL
