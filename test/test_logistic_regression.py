import ml_mini.logistic_model as logistic_model
import numpy as np

def test_logistic_regression(X, y, method):  
   model = logistic_model.LogisticRegression()
   model.fit(X, y, learning_rate=0.01, max_iter=1000, method=method)

   print("Coefficients:", model.coef_)
   print("Number of iterations:", model.n_iterations_)

   y_pred_proba = model.predict_proba(X)
   y_pred = model.predict(X)

   print("Predicted probabilities:", y_pred_proba)
   print("Predicted classes:", y_pred)

def test_perceptron(X, y):
   model = logistic_model.Perceptron()
   model.fit(X, y, learning_rate=0.01, n_iterations=100)

   print("Coefficients:", model.coef_)

   y_pred = model.predict(X)

   print("Predicted classes:", y_pred)

if __name__ == "__main__":
   X = np.array([
         [0, 0],
         [1, 0],
         [0, 1],
         [1, 1],
         [2, 1],
         [1, 2],
         [2, 2],
         [3, 1]
      ], dtype=float)

   true_coef = np.array([2.0, -1.0])
   true_intercept = -1.0

   linear_output = X @ true_coef + true_intercept
   y = (linear_output >= 0).astype(int)

   print("===== Logistic Regression Test =====")

   print("True Coefficients:", true_coef)
   print("True Intercept:", true_intercept)

   print("\nTesting Batch Gradient Descent:")
   test_logistic_regression(X, y, method='batch')
   print("\nTesting Stochastic Gradient Descent:")
   test_logistic_regression(X, y, method='stochastic')

   print("\n===== Perceptron Test =====")
   test_perceptron(X, y)
