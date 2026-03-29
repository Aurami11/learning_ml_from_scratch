import ml_mini.linear_model as linear_model
import numpy as np
import matplotlib.pyplot as plt

def test_linear_regression(X, y, method):

    model = linear_model.LinearRegression()
    model.fit(X, y, learning_rate=0.01, n_iterations=3000, method=method)

    print("Coefficients:", model.coef_)
    print("Number of iterations:", model.n_iterations_)

    y_pred = model.predict(X)

    plt.plot(model.loss_history_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.show()

    plt.scatter(y, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.show()

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
   true_intercept = 5.0

   y = X @ true_coef + true_intercept

   print("True Coefficients:", true_coef)
   print("True Intercept:", true_intercept)

   print("\nTesting Batch Gradient Descent:")
   test_linear_regression(X, y, method='batch')
   print("\nTesting Stochastic Gradient Descent:")
   test_linear_regression(X, y, method='stochastic')