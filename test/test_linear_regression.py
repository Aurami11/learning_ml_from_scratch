import ml_mini.linear_model as linear_model
import numpy as np
import matplotlib.pyplot as plt

def test_linear_regression():
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

    model = linear_model.LinearRegression()
    model.fit(X, y, learning_rate=0.01, n_iterations=3000)

    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
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

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert len(model.loss_history_) > 0
    assert model.n_iterations_ is not None

    assert np.allclose(model.coef_, true_coef, atol=1e-1)
    assert np.isclose(model.intercept_, true_intercept, atol=1e-1)
    assert np.allclose(y_pred, y, atol=1e-1)
    assert model.loss_history_[0] > model.loss_history_[-1]

if __name__ == "__main__":
    test_linear_regression()
    print("All tests passed!")