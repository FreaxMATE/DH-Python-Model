"""
Unit tests for dhmodel package.

Tests the core functionality of the DH-model Python implementation.
"""

import unittest
import numpy as np
import pandas as pd
from dhmodel import daylength, run_dh_model, load_dh_model_ins


class TestDaylength(unittest.TestCase):
    """Test cases for the daylength function."""
    
    def test_daylength_calculation(self):
        """Test daylength calculation returns valid values."""
        day = np.arange(1, 366)
        sangle = -23.45 * np.pi/180 * np.cos((np.pi/180) * (day + 10))
        dayl = daylength(sangle, LAT=48.6542)
        
        # Check that daylength is between 0 and 24 hours
        self.assertTrue(np.all(dayl >= 0))
        self.assertTrue(np.all(dayl <= 24))
    
    def test_daylength_seasonal_pattern(self):
        """Test that summer has longer days than winter."""
        day = np.arange(1, 366)
        sangle = -23.45 * np.pi/180 * np.cos((np.pi/180) * (day + 10))
        dayl = daylength(sangle, LAT=48.6542)
        
        # Summer (days 150-210) should have longer days than winter (days 0-60)
        summer_avg = np.mean(dayl[150:210])
        winter_avg = np.mean(dayl[0:60])
        self.assertGreater(summer_avg, winter_avg)
    
    def test_daylength_scalar_input(self):
        """Test daylength with scalar input."""
        sangle = 0.0  # Equinox
        dayl = daylength(sangle, LAT=48.6542)
        
        # Should be close to 12 hours at equinox
        self.assertIsInstance(dayl, float)
        self.assertAlmostEqual(dayl, 12.0, delta=1.0)
    
    def test_daylength_extreme_latitudes(self):
        """Test daylength at extreme latitudes."""
        day = np.arange(1, 366)
        sangle = -23.45 * np.pi/180 * np.cos((np.pi/180) * (day + 10))
        
        # High latitude (near Arctic)
        dayl_high = daylength(sangle, LAT=65.0)
        self.assertTrue(np.all(dayl_high >= 0))
        self.assertTrue(np.all(dayl_high <= 24))
        
        # Low latitude (tropical)
        dayl_low = daylength(sangle, LAT=10.0)
        self.assertTrue(np.all(dayl_low >= 0))
        self.assertTrue(np.all(dayl_low <= 24))


class TestLoadData(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Load example data."""
        self.data = load_dh_model_ins()
    
    def test_data_structure(self):
        """Test that loaded data has correct structure."""
        self.assertIn('Inputs_GRA_3years', self.data)
        self.assertIn('Inputs_ABR_4years', self.data)
        self.assertIn('Parameters', self.data)
    
    def test_parameters_structure(self):
        """Test that parameters have correct structure."""
        params = self.data['Parameters']
        self.assertIn('best', params)
        self.assertIn('upper', params)
        self.assertIn('lower', params)
        
        # Check parameter dimensions
        self.assertEqual(len(params['best']), 10)
        self.assertEqual(len(params['upper']), 10)
        self.assertEqual(len(params['lower']), 10)
    
    def test_forcing_data_columns(self):
        """Test that forcing data has required columns."""
        gra_data = self.data['Inputs_GRA_3years']
        required_cols = ['Tair', 'SW', 'Cpool', 'week', 'year.datetime.']
        
        for col in required_cols:
            self.assertIn(col, gra_data.columns)
    
    def test_forcing_data_values(self):
        """Test that forcing data has reasonable values."""
        gra_data = self.data['Inputs_GRA_3years']
        
        # Temperature should be reasonable
        self.assertTrue(gra_data['Tair'].min() > -50)
        self.assertTrue(gra_data['Tair'].max() < 50)
        
        # Soil water should be between 0 and 1 (or reasonable range)
        self.assertTrue(gra_data['SW'].min() >= 0)
        self.assertTrue(gra_data['SW'].max() <= 2)
        
        # Cpool should be positive
        self.assertTrue(gra_data['Cpool'].min() >= 0)


class TestRunDHModel(unittest.TestCase):
    """Test cases for the run_dh_model function."""
    
    def setUp(self):
        """Load example data for testing."""
        self.data = load_dh_model_ins()
        self.gra_inputs = self.data['Inputs_GRA_3years']
        self.params_best = self.data['Parameters']['best']
    
    def test_model_runs_without_error(self):
        """Test that model runs without errors."""
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_model_output_structure(self):
        """Test that model output has correct structure."""
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        # Check required columns
        self.assertIn('Dens', result.columns)
        self.assertIn('Incr', result.columns)
        self.assertIn('Nr', result.columns)
        self.assertIn('weeks', result.columns)
        self.assertIn('years', result.columns)
    
    def test_model_output_length(self):
        """Test that output has same length as input."""
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        self.assertEqual(len(result), len(self.gra_inputs))
    
    def test_increment_non_negative(self):
        """Test that increment values are non-negative."""
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        self.assertTrue(np.all(result['Incr'].values >= 0))
    
    def test_cell_numbers_non_negative(self):
        """Test that cell numbers are non-negative."""
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        self.assertTrue(np.all(result['Nr'].values >= 0))
    
    def test_model_with_daylength(self):
        """Test model with daylength constraint."""
        # Create daylength vector
        day = np.arange(1, len(self.gra_inputs) + 1)
        sangle = -23.45 * np.pi/180 * np.cos((np.pi/180) * (day + 10))
        dayl = daylength(sangle, LAT=48.6542)
        
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            dayl=dayl,
            DH_plot=False
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.gra_inputs))
    
    def test_model_with_volumetric_sw(self):
        """Test model with volumetric soil water conversion."""
        result = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            Rw_vol=True,
            DH_plot=False
        )
        
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_cold_temperature_stops_growth(self):
        """Test that cold temperatures prevent growth."""
        # Create forcing with all cold temperatures
        n_weeks = 10
        cold_temp = np.zeros(n_weeks)  # 0°C (below Tmin=5)
        rw = np.full(n_weeks, 0.5)
        cpool = np.full(n_weeks, 2.0)
        weeks = np.arange(1, n_weeks + 1)
        years = np.full(n_weeks, 2020)
        
        result = run_dh_model(
            Tair=cold_temp,
            Rw=rw,
            Cpool=cpool,
            params=self.params_best,
            week=weeks,
            year=years,
            DH_plot=False
        )
        
        # No growth should occur
        self.assertTrue(np.all(result['Incr'].values == 0))
        self.assertTrue(np.all(result['Nr'].values == 0))
    
    def test_dry_conditions_stop_growth(self):
        """Test that dry conditions prevent growth."""
        # Create forcing with dry conditions
        n_weeks = 10
        temp = np.full(n_weeks, 20.0)  # Warm enough
        rw = np.zeros(n_weeks)  # Dry (below Rwmin=0.15)
        cpool = np.full(n_weeks, 2.0)
        weeks = np.arange(1, n_weeks + 1)
        years = np.full(n_weeks, 2020)
        
        result = run_dh_model(
            Tair=temp,
            Rw=rw,
            Cpool=cpool,
            params=self.params_best,
            week=weeks,
            year=years,
            DH_plot=False
        )
        
        # No growth should occur
        self.assertTrue(np.all(result['Incr'].values == 0))
        self.assertTrue(np.all(result['Nr'].values == 0))


class TestModelConsistency(unittest.TestCase):
    """Test consistency between different model runs."""
    
    def setUp(self):
        """Load example data."""
        self.data = load_dh_model_ins()
        self.gra_inputs = self.data['Inputs_GRA_3years']
        self.params_best = self.data['Parameters']['best']
    
    def test_reproducibility(self):
        """Test that model produces same results with same inputs."""
        result1 = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        result2 = run_dh_model(
            Tair=self.gra_inputs['Tair'],
            Rw=self.gra_inputs['SW'],
            Cpool=self.gra_inputs['Cpool'],
            params=self.params_best,
            week=self.gra_inputs['week'],
            year=self.gra_inputs['year.datetime.'],
            DH_plot=False
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result1['Incr'].values, result2['Incr'].values)
        np.testing.assert_array_equal(result1['Nr'].values, result2['Nr'].values)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
