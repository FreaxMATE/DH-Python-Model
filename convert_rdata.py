"""
Convert R data file to Python pickle format.

This script reads the DH_model_ins.rda file from the R package
and converts it to a pickle file for use in Python.
"""

import pickle
import pandas as pd
import numpy as np

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import os
    
    # Load the R data file
    r_data_path = '../DH-model/data/DH_model_ins.rda'
    print(f"Loading R data from: {r_data_path}")
    ro.r['load'](r_data_path)
    
    # Get the loaded object
    dh_model_ins_r = ro.r['DH_model_ins']
    
    # Get the names of the list
    names = list(dh_model_ins_r.names)
    print(f"Found data components: {names}")
    
    result = {}
    
    # Process each component
    for i, name in enumerate(names):
        element_raw = dh_model_ins_r[i]  # Get raw R object
        
        if name == 'Parameters':
            # Handle nested list for Parameters - stay outside conversion context
            param_names = list(element_raw.names)
            result[name] = {}
            for j, param_name in enumerate(param_names):
                result[name][param_name] = np.array(element_raw[j])
            print(f"  Parameters: {param_names}")
        else:
            # Convert R dataframe to pandas using context manager
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.get_conversion().rpy2py(element_raw)
                result[name] = df
                print(f"  {name}: {len(df)} rows, columns: {list(df.columns)}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save as pickle
    output_path = 'data/DH_model_ins.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\n✓ Successfully converted R data to {output_path}")
    print(f"  Keys: {list(result.keys())}")

except ImportError:
    print("ERROR: rpy2 is not installed.")
    print("\nTo install rpy2:")
    print("  pip install rpy2")
    print("\nOr if using nix, add 'python3Packages.rpy2' to your flake.nix")
    
except Exception as e:
    print(f"ERROR: {e}")
    print("\nAlternative: You can use the synthetic example data (it works fine for testing)")
