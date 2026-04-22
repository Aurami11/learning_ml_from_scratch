"""
Preprocessing utilities for machine learning.
"""

import numpy as np

def X_bias(X):
   "Add a bias term (intercept) to the input features"
   n_samples = X.shape[0]
   bias = np.ones((n_samples, 1))
   return np.hstack((bias, X))