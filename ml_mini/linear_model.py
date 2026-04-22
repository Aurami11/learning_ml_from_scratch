import numpy as np
from ml_mini.utils.preprocessing import X_bias

class LinearRegression:
   def __init__(self):
         self.coef_ = None
         self.loss_history_ = []
         self.n_iterations_ = None
         self.gradient_method = None
   def _generate_coef(self, n_features):
        "Initialize coefficients and intercept with random values"
        np.random.seed(42)  # For reproducibility
        self.coef_ = np.random.rand(n_features)

   def predict(self, X, add_bias=True):
        "Predict linear regression output given input X"

        if add_bias:
            X = X_bias(X)
        return X @ self.coef_
   
   def _batch_gradient_descent(self, X, y, learning_rate, max_iter, eps=1e-6, _start_coef=None, add_bias=True):
      """Batch Gradient Descent reduces the loss on the entire dataset"""

      if add_bias:
         X = X_bias(X)
      n_features = X.shape[1]

      if _start_coef is not None:
         self.coef_ = _start_coef
      else:
         self._generate_coef(n_features)

      for i in range(max_iter):
         y_pred = self.predict(X, add_bias=False)

         loss_gradient = X.T @ (y_pred - y)

         self.coef_ -= learning_rate * loss_gradient

         self.loss_history_.append(np.mean((y_pred - y) ** 2))

         if i > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < eps:
            self.n_iterations_ = i
            break
         
   def _stochastic_gradient_descent(self, X, y, learning_rate, _start_coef=None, _start_intercept=None, add_bias=True):
      """Stochastic Gradient Descent updates the coefficients for each training example"""

      if add_bias:
         X = X_bias(X)
      n_features = X.shape[1]

      if _start_coef is not None and _start_intercept is not None:
         self.coef_ = _start_coef
      else:
         n_features = X.shape[1]
         self._generate_coef(n_features)
    
      for x_i, y_i in zip(X, y):
         y_pred = self.predict(x_i.reshape(1, -1), add_bias=False)
         loss_gradient = (y_pred - y_i) * x_i

         self.coef_ -= learning_rate * loss_gradient
   
   def fit(self, X, y, learning_rate=0.01, n_iterations=1000, eps=1e-6, method='batch'):
      if method == 'batch':
         self._batch_gradient_descent(X, y, learning_rate, n_iterations, eps)
         self.gradient_method = 'batch'
      
      elif method == "stochastic":
         self._stochastic_gradient_descent(X, y, learning_rate)
         self.gradient_method = 'stochastic'

      else:
         raise ValueError("Invalid method. Use 'batch' or 'stochastic'.")

   def refit(self, X, y, learning_rate=0.01):
      "Refit the model with the new data"
      if self.gradient_method == 'batch':
         self._batch_gradient_descent(X, y, learning_rate, max_iter=1000, _start_coef=self.coef_)

      elif self.gradient_method == 'stochastic':
         self._stochastic_gradient_descent(X, y, learning_rate, _start_coef=self.coef_)
      else:
         raise ValueError("Model has not been fitted yet. Please call fit() before refit().")


class LocallyWeightedLinearRegression(LinearRegression):
   def __init__(self, tau=1.0):
      self.tau = tau
      self.coef_ = None
      self.loss_history_ = []
      self.n_iterations_ = None

   def _gaussian_kernel(self, x, X):
      distances = np.linalg.norm(X - x, axis=1)
      weights = np.exp(- (distances ** 2) / (2 * self.tau ** 2))
      return weights
   
   def predict(self, X, add_bias=True):
      if add_bias:
         X = X_bias(X)

      return X @ self.coef_
   
   def fit(self, x, X, y, learning_rate=0.01, max_iter=1000, eps=1e-6):
      """Fit the model using locally weighted linear regression for a single query point x"""

      X = X_bias(X)
      x = X_bias(x.reshape(1, -1)).flatten()

      self._generate_coef(X.shape[1])

      for i in range(max_iter):
         y_pred = self.predict(X, add_bias=False)
         linear_weighted_loss = (y_pred - y) * self._gaussian_kernel(x, X)

         loss_gradient = X.T @ linear_weighted_loss

         self.coef_ -= learning_rate * loss_gradient
         self.loss_history_.append(np.mean(linear_weighted_loss ** 2))
         if i > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < eps:
            self.n_iterations_ = i
            break
