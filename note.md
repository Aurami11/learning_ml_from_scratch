# Notes

### An Introduction to Statistical Learning — James, Witten, Hastie, Tibshirani

Chapter 2 Statistical Learning is great to fix the basic knowledge statistical learning.

- Predictions : Predict y = f(X) where the predictor f is "treated as a black box" and only, matter, the prediction accuracy.
  - reducible error : depend on how well defined is the model
  - irreducible error : error due to intricate variability of the target or effect of missing variable
- Inference : Understanding the relationship between X and y
- Parametric model & non parametric model
- etc...

### Stanford CS229 Notes

#### Linear Reg

- Batch Gradient Descent : Try to minimize the loss function by reducing the loss on all the training data. It's good for a small dataset but as the dataset get bigger, it become computationally heavy.
- Stochastic Gradient Descent : Try to minimize the loss function by minimizing the loss on each training data. Because, here we approach the problem in an incremental way, it's less heavy on bigger dataset and it's also easy to add new training data without having to retrain the model.