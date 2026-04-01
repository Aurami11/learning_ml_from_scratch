import matplotlib.pyplot as plt
import numpy as np
from ml_mini.linear_model import LocallyWeightedLinearRegression

def test_locally_weighted_regression(x, X, y, tau):
   x = x.reshape(1, -1)  # 2D for prediction

   model = LocallyWeightedLinearRegression(tau=tau)
   model.fit(x, X, y, learning_rate=0.01, max_iter=1000)

   print("Locally Weighted Coefficients:", model.coef_)

   y_pred = model.predict(x)

   print(f"Predicted value for query point {x}: {y_pred[0]}")
   print(f"Actual value for query point {x}: {x[0][0] ** 2 + 2 * x[0][1] - x[0][1] + 3}")

   # Loss history plot
   plt.plot(model.loss_history_)
   plt.xlabel("Iteration")
   plt.ylabel("Loss")
   plt.title(f"Loss History (tau={tau})")
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

   y = X[:, 0] ** 2 + 2 * X[:, 0] - X[:, 1] + 3  # Non-linear relationship

   x_query = np.array([[1.5, 1.5], [2.5, 0.5], [0.5, 2.5]])
   tau_value = [1.0, 0.5, 1.1]
   for x, tau in zip(x_query, tau_value):
      print(f"\nTesting Locally Weighted Linear Regression for query point {x} with tau={tau}:")
      test_locally_weighted_regression(x, X, y, tau=tau)