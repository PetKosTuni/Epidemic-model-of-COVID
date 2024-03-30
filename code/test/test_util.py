from contextlib import AbstractContextManager
from typing import Any
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

if __name__ == '__main__':
    unittest.main()