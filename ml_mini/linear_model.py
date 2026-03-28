import numpy as np

class LinearRegression:
   def __init__(self):
         self.coef_ = None
         self.intercept_ = None
         self.loss_history_ = []
         self.n_iterations_ = None
   def _generate_coef(self, n_features):
        "Initialize coefficients and intercept with random values"
        self.coef_ = np.random.rand(n_features)
        self.intercept_ = np.random.rand(1)

   def predict(self, X):
        "Predict linear regression output given input X"
        return X @ self.coef_ + self.intercept_
   
   def _batch_gradient_descent(self, X, y, learning_rate, max_iter, eps=1e-6):
      """Batch Gradient Descent reduces the loss on the entire dataset"""

      n_features = X.shape[1]
      self._generate_coef(n_features)

      for i in range(max_iter):
         y_pred = self.predict(X)

         loss_gradient_coef = X.T @ (y_pred - y)
         loss_gradient_intercept = np.sum(y_pred - y) # (y_pred - y).T @ np.ones(X.shape[0])

         self.coef_ -= learning_rate * loss_gradient_coef
         self.intercept_ -= learning_rate * loss_gradient_intercept

         self.loss_history_.append(np.mean((y_pred - y) ** 2))

         if i > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < eps:
            self.n_iterations_ = i
            break
         
   def _stochastic_gradient_descent(self, X, y, learning_rate, max_iter, eps=1e-6):
       return None
   
   def fit(self, X, y, learning_rate=0.01, n_iterations=1000, eps=1e-6, method='batch'):
      if method == 'batch':
         self._batch_gradient_descent(X, y, learning_rate, n_iterations, eps)
      
      elif method == "stochastic":
         self._stochastic_gradient_descent(X, y, learning_rate, n_iterations, eps)