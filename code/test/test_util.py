import unittest
import os
import numpy as np
os.environ['DC_STATEHOOD'] = '1'

from util import *

# This testing file contains unit tests for the utilities in util.py.

class TestUtilities(unittest.TestCase):

  def test_ensure_float64(self):
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = ensure_float64(arr)
    self.assertIs(result.dtype, np.dtype(np.float64))
    arr = np.array([1, 2, 3])
    result = ensure_float64(arr)
    self.assertIs(result.dtype, np.dtype(np.float64))
    arr = np.array(['1.0', '2.0', '3.0'])
    result = ensure_float64(arr)
    self.assertIs(result.dtype, np.dtype(np.float64))
    arr = np.array([1.0, 2.0, 3.0], dtype=object)
    result = ensure_float64(arr)
    self.assertIs(result.dtype, np.dtype(np.float64))
    arr = np.array([])
    result = ensure_float64(arr)
    self.assertIs(result.dtype, np.dtype(np.float64))
    arr = np.array(['moi', 1.0, 2.0])
    with self.assertRaises(ValueError):
      ensure_float64(arr)
  
  def test_get_county_list(self):
    counties = get_county_list()
    self.assertTrue(len(counties) > 0)
    self.assertEqual((len(counties)), 991)
    self.assertIsInstance(counties, list)
    self.assertTrue(all(isinstance(county, str) for county in counties))
    self.assertIn('Coffee_Tennessee', counties)
    self.assertIn('Calumet_Wisconsin', counties)
    self.assertIn('Chippewa_Wisconsin', counties)
    self.assertIn('Columbia_Wisconsin', counties)
    self.assertIn('Dane_Wisconsin', counties)
    self.assertNotIn("California", counties)
    self.assertNotIn("Texas", counties)
    self.assertNotIn("Brazil", counties)

  def test_num2str(self):
    self.assertEqual(num2str(11), '11000')
    self.assertEqual(num2str(222), '22200')
    self.assertEqual(num2str(4444), '44440')

  def test_state2fips(self):
    self.assertEqual(state2fips("Alaska"), "02")
    self.assertEqual(state2fips("California"), "06")
    self.assertEqual(state2fips("Texas"), "48")
    self.assertEqual(state2fips("West Virginia"), "54")
    self.assertEqual(state2fips("New York"), "36")
    self.assertEqual(state2fips("US"), "US")
    with self.assertRaises(AttributeError):
      state2fips("Finland")

  def test_lognorm_ave(self):
    x = lognorm_ave([1.5,1,1,1,1,1,1,1,1])
    self.assertAlmostEqual(x, 1.020157, 6)
    x1 = lognorm_ave([1.5,1,1,1,1,1,1,1,1,2,2,2])
    self.assertAlmostEqual(x1, 1.241452, 6)
    x2 = lognorm_ave([1,2,3,4,5,6,7,8,9], 3, 0.511)
    self.assertAlmostEqual(x2, 5.8755, 4)
    x3 = lognorm_ave([])
    self.assertEqual(x3, 0)

  def test_first_valid_date(self):
    dates = ["1", "hello", "01/01/01/01", "2000/01", "invalid date",
             "01/01/23", "12/31/24"]
    expected_index = 5
    result = first_valid_date(dates)
    self.assertEqual(result, expected_index)
    dates = ["1", "hello", "01/01/01/01", "2000/01", "invalid date",
             "01/01/2023", "12/31/2024"]
    expected_index = 5
    result = first_valid_date(dates, format="%m/%d/%Y")
    self.assertEqual(result, expected_index)
    dates = ["01/01/2023", "12/31/2024"]
    expected_index = 0
    result = first_valid_date(dates, format="%m/%d/%Y")
    self.assertEqual(result, expected_index)
    dates = ["1", "2", "01-01-23", "12-31-24"]
    expected_index = 2
    result = first_valid_date(dates, format="%m-%d-%y")
    self.assertEqual(result, expected_index)
    dates = ["1", "2", "3"]
    expected_index = len(dates)
    result = first_valid_date(dates)
    self.assertEqual(result, expected_index)
    dates = []
    expected_index = 0 
    result = first_valid_date(dates)
    self.assertEqual(result, expected_index)

  def test_func(self):
    coefficient1 = func(0.1, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient1, 0.1323745353, 9)
    coefficient2 = func(0, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient2, 0.1357256127, 9)
    coefficient3 = func(0, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient3, 0)
    coefficient4 = func(0.1, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient4, 0)
    coefficient5 = func(0.1, 0.15, 0, 0.4)
    self.assertEqual(coefficient5, 0.15)
    coefficient6 = func(0.1, 0.15, 0.25, 0)
    self.assertAlmostEqual(coefficient6, 0.1462964868, 9)
    coefficient7 = func(0,0,0,0)
    self.assertEqual(coefficient7, 0)

  def test_func_root(self):
    coefficient1 = func_root(0.1, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient1, 0.1385980964, 9)
    coefficient2 = func_root(0, 0.15, 0.25, 0.4)
    self.assertEqual(coefficient2, 0.15)
    coefficient3 = func_root(0, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient3, 0)
    coefficient4 = func_root(0.1, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient4, 0)
    coefficient5 = func_root(0.1, 0.15, 0, 0.4)
    self.assertEqual(coefficient5, 0.15)
    coefficient6 = func_root(0.1, 0.15, 0.25, 0)
    self.assertAlmostEqual(coefficient6, 0.1385980964, 9)
    coefficient7 = func_root(0,0,0,0)
    self.assertEqual(coefficient7, 0)

  def test_func_new(self):
    coefficient1 = func_new(0.1, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient1,  0.332374535, 9)
    coefficient2 = func_new(0, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient2, 0.33572561270, 9)
    coefficient3 = func_new(0, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient3, 0.2)
    coefficient4 = func_new(0.1, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient4, 0.2)
    coefficient5 = func_new(0.1, 0.15, 0, 0.4)
    self.assertEqual(coefficient5, 0.35)
    coefficient6 = func_new(0.1, 0.15, 0.25, 0)
    self.assertAlmostEqual(coefficient6, 0.346296486804, 9)
    coefficient7 = func_new(0,0,0,0)
    self.assertEqual(coefficient7, 0.2)
  
  def test_func_poly(self):
    coefficient1 = func_poly(0.1, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient1, 0.1355403005, 9)
    coefficient2 = func_poly(0, 0.15, 0.25, 0.4)
    self.assertAlmostEqual(coefficient2, 0.13789840728, 9)
    coefficient3 = func_poly(0, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient3, 0)
    coefficient4 = func_poly(0.1, 0.0, 0.25, 0.4)
    self.assertEqual(coefficient4, 0)
    coefficient5 = func_poly(0.1, 0.15, 0, 0.4)
    self.assertEqual(coefficient5, 0.15)
    coefficient6 = func_poly(0.1, 0.15, 0.25, 0)
    self.assertAlmostEqual(coefficient6, 0.146468113451, 9)
    coefficient7 = func_poly(0,0,0,0)
    self.assertEqual(coefficient7, 0)

  def test_func_sin(self):
    coefficient1 = func_sin(0.1, 0.15, 0.25)
    self.assertAlmostEqual(coefficient1, 0.04635254915, 9)
    coefficient2 = func_sin(0, 0.15, 0.25)
    self.assertAlmostEqual(coefficient2,  0.0333781400, 9)
    coefficient3 = func_sin(0, 0.0, 0.25)
    self.assertEqual(coefficient3, 0)
    coefficient4 = func_sin(0.1, 0.0, 0.25)
    self.assertEqual(coefficient4, 0)
    coefficient5 = func_sin(0.1, 0.15, 0)
    self.assertAlmostEqual(coefficient5, 0.013445896335, 9)
    coefficient6 = func_sin(0.1, 0.15, 0.25)
    self.assertAlmostEqual(coefficient6, 0.046352549156, 9)
    coefficient7 = func_sin(0,0,0)
    self.assertEqual(coefficient7, 0)

  def test_write_val_to_json(self):
    input_path = "test/testinput"
    output_path = "test/testoutput"
    test_file1 = "util_test_output1"
    test_file2 = "util_test_output2"
    error_msg = "Test output file not found!"
    with open(os.path.join(input_path, "testinput_util.json"), "r") as test_file:
      data = json.load(test_file)
    write_val_to_json(data, os.path.join(output_path, test_file1),
                      os.path.join(output_path, test_file2), limit=0.5e-5)
    with open(os.path.join(output_path,test_file2), "r") as test_output:
      data2 = json.load(test_output)
    self.assertTrue(os.path.exists(os.path.join(output_path, test_file2)),
                    error_msg)
    self.assertTrue(os.path.exists(os.path.join(output_path, test_file1)),
                    error_msg)
    self.assertLess(abs(data2["Alabama"][0] - 980637.0), 10)
    self.assertLess(abs(data2["Alabama"][1] - 17829), 10)
    self.assertLess(abs(data2["Alabama"][3] - 575091), 10)

if __name__ == '__main__':
    unittest.main()
