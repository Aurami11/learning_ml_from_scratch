"""
This module implements the Logistic Regression algorithm for classification tasks, as well as a simple Perceptron model. 
The Perceptron class implements a basic perceptron learning algorithm for binary classification.
The implementation still lack the Newton-Raphson method and multiclass classification support, which can be added in future iterations.
Should also implement boundary decision visualization for 2D datasets.
"""

import numpy as np
from ml_mini.linear_model import LinearRegression
from ml_mini.utils.preprocessing import X_bias

class LogisticRegression(LinearRegression) :
   def __init__(self) :
      self.coef_ = None
      self.loss_history_ = []
      self.n_iterations_ = None
      self.gradient_method = None

   def _sigmoid(self, z) :
      return 1 / (1 + np.exp(-z))
   
   def predict_proba(self, X, add_bias=True) :
      if add_bias :
         X = X_bias(X)
      linear_output = X @ self.coef_
      return self._sigmoid(linear_output)
   
   def predict(self, X, add_bias=True, threshold=0.5) :
      proba = self.predict_proba(X, add_bias)
      return (proba >= threshold).astype(int)
   
   def fit(self, X, y, learning_rate=0.01, max_iter=1000, eps=1e-6, method='batch', add_bias=True) :
      
      # The fit method is similar to LinearRegression but uses the logistic loss and gradient
      if method == 'batch' :
         self._batch_gradient_descent(X, y, learning_rate, max_iter, eps, add_bias=add_bias)
         self.gradient_method = 'batch'
      elif method == 'stochastic' :
         self._stochastic_gradient_descent(X, y, learning_rate, add_bias=add_bias)
         self.gradient_method = 'stochastic'

   def _newton_raphson(self, X, y, max_iter=1000, eps=1e-6) :
      """Newton-Raphson method for logistic regression"""
      X = X_bias(X)
      #TODO: Implement Newton-Raphson method for logistic regression
      pass


class Perceptron(LinearRegression) :
   def __init__(self) :
      self.coef_ = None

   def predict(self, X, add_bias=True) :
      if add_bias :
         X = X_bias(X)
      linear_output = X @ self.coef_
      return (linear_output >= 0).astype(int)
   
   def fit(self, X, y, learning_rate=0.01, n_iterations=1000, add_bias=True) :
      if add_bias :
         X = X_bias(X)

      n_features = X.shape[1]
      self.coef_ = np.zeros(n_features)

      for i in range(n_iterations) :
         for x_i, y_i in zip(X, y) :
            y_pred = self.predict(x_i.reshape(1, -1), add_bias=False)
            update = learning_rate * (y_i - y_pred[0])
            self.coef_ += update * x_i

   