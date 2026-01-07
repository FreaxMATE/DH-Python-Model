# DH-model (Python Version)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**Deleuze et Houllier 1998 model with some modifications**

A process-based xylem growth model for simulating wood density and ring increment. This is a Python version of the original R package available at https://github.com/teatree1212/DH-model.

## Features

- 🌲 Process-based tree ring growth simulation
- 📊 Wood density and increment calculations
- 🌡️ Temperature, soil moisture, and carbon pool responses
- 📈 Built-in visualization tools
- 🧪 Comprehensive test suite
- 📦 Easy-to-use Python package

## Installation

### From source (recommended for development)

```bash
cd dhmodel-py
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from dhmodel import load_dh_model_ins, run_dh_model

# Load example data and parameters
data = load_dh_model_ins()

# Run the DH-model
result = run_dh_model(
    Tair=data['Inputs_GRA_3years']['Tair'],
    Rw=data['Inputs_GRA_3years']['SW'],
    Cpool=data['Inputs_GRA_3years']['Cpool'],
    params=data['Parameters']['best'],
    week=data['Inputs_GRA_3years']['week'],
    year=data['Inputs_GRA_3years']['year.datetime.'],
    DH_plot=True  # Show diagnostic plots
)

# Access results
print(result[['Dens', 'Incr', 'Nr']].head())
```
## Usage Examples

### Basic Model Run

```python
import numpy as np
from dhmodel import run_dh_model

# Prepare forcing data
weeks = 53
temperature = np.random.uniform(10, 25, weeks)  # °C
soil_moisture = np.random.uniform(0.3, 0.7, weeks)  # relative
carbon_pool = np.random.uniform(1.5, 2.5, weeks)  # Kg[C]
week_numbers = np.arange(1, weeks + 1)
years = np.full(weeks, 2020)

# Define parameters [Tmin, Rwmin, D_n_max, b, D_v_max, delta, chi, R_w_crit, D_m_max, dayl_min]
params = np.array([5.0, 0.15, 0.8, 0.15, 0.0052, 0.5, 4.0, 0.4, 0.0045, 12.0])

# Run model
result = run_dh_model(
    Tair=temperature,
    Rw=soil_moisture,
    Cpool=carbon_pool,
    params=params,
    week=week_numbers,
    year=years,
    DH_plot=True
)
```

### Using Daylength Constraint

```python
from dhmodel import daylength, run_dh_model
import numpy as np

# Calculate daylength for a year
day = np.arange(1, 366)
sangle = -23.45 * np.pi/180 * np.cos((np.pi/180) * (day + 10))
dayl = daylength(sangle, LAT=48.6542)  # Latitude in degrees

# Run model with daylength constraint
result = run_dh_model(
    Tair=temperature,
    Rw=soil_moisture,
    Cpool=carbon_pool,
    params=params,
    week=week_numbers,
    year=years,
    dayl=dayl[:weeks],  # Use first weeks of daylength
    DH_plot=True
)
```

### Helper Functions

```python
from dhmodel.utils import calculate_height_from_diameter, aggregate_by_year
import numpy as np

# Convert diameter to height using allometric equation
diameter_cm = np.cumsum([0.5, 0.8, 1.2, 0.9])  # Cumulative growth
height_m = calculate_height_from_diameter(diameter_cm, a=1.2, b=0.6)

# Aggregate climate data by year
import pandas as pd
forcing = pd.read_csv('climate_data.csv')
annual_temp = aggregate_by_year(forcing, 'Malmö', 'Temp')
```

## Model Parameters

The model requires 10 parameters:

| Parameter | Description | Typical Value | Unit |
|-----------|-------------|---------------|------|
| `Tmin` | Minimum temperature for growth | 5.0 | °C |
| `Rwmin` | Minimum relative soil moisture | 0.15 | - |
| `D_n_max` | Maximum cell production rate | 0.8 | cells/week |
| `b` | Temperature response parameter | 0.15 | - |
| `D_v_max` | Maximum volume increment | 0.0052 | m³/week |
| `delta` | Carbon response parameter | 0.5 | - |
| `chi` | Soil moisture adjustment | 4.0 | - |
| `R_w_crit` | Critical soil moisture | 0.4 | - |
| `D_m_max` | Maximum mass increment | 0.0045 | kg/week |
| `dayl_min` | Minimum daylength for growth | 12.0 | hours |

## Model Output

The model returns a pandas DataFrame with:

- **`Dens`**: Wood density (g/cm³)
- **`Incr`**: Ring increment (mm/week)
- **`Nr`**: Number of cells produced
- **`weeks`**: Week number
- **`years`**: Year

## Testing

Run the test suite to verify installation:

```bash
cd dhmodel-py
python -m pytest tests/test_dhmodel.py -v
```

Or using unittest:

```bash
python -m unittest tests/test_dhmodel.py
```

## Related Resources & References

### Main Model

The main principles and equations are from:

- **Deleuze, C., & Houllier, F. (1998).** A Simple Process-based Xylem Growth Model for Describing Wood Microdensitometric Profiles. *Journal of Theoretical Biology*, 193(1), 99–113. [https://doi.org/10.1006/jtbi.1998.0689](https://doi.org/10.1006/jtbi.1998.0689)

### Parameters & Modifications

Parameters are mostly from:

- **Wilkinson, S., et al. (2015).** Biophysical modelling of intra-ring variations in tracheid features and wood density of Pinus pinaster trees exposed to seasonal droughts. *Tree Physiology*, 35(3), 305–318. [https://doi.org/10.1093/treephys/tpv010](https://doi.org/10.1093/treephys/tpv010)

- **Cuny, H. E., & Rathgeber, C. B. K. (2016).** Xylogenesis: Coniferous Trees of Temperate Forests Are Listening to the Climate Tale during the Growing Season But Only Remember the Last Words! *Plant Physiology*, 171(1), 306–317. [https://doi.org/10.1104/pp.16.00037](https://doi.org/10.1104/pp.16.00037)

### Daylength Function

Based on the BASFOR model by David Cameron and Marcel van Oijen.

## Differences from R Version

This Python implementation maintains identical functionality to the R package while following Python conventions:

- Uses NumPy arrays instead of R vectors
- Uses pandas DataFrames instead of R data.frames
- Uses matplotlib for plotting instead of base R graphics
- Follows PEP 8 style guidelines
- Includes comprehensive docstrings and type hints

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

## License

Mozilla Public License Version 2.0 (MPL-2.0)

## Author

**Annemarie Eckes-Shephard**

Python conversion by: AI Assistant (2026)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this model in your research, please cite:

```bibtex
@article{deleuze1998simple,
  title={A Simple Process-based Xylem Growth Model for Describing Wood Microdensitometric Profiles},
  author={Deleuze, C and Houllier, F},
  journal={Journal of Theoretical Biology},
  volume={193},
  number={1},
  pages={99--113},
  year={1998},
  publisher={Elsevier}
}
```

## Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Note:** This is a Python port of the original R package. For the R version, please see the `DH-model/` directory.
