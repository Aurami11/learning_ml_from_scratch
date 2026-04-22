# Notes

### An Introduction to Statistical Learning — James, Witten, Hastie, Tibshirani

Chapter 2 Statistical Learning is great to fix the basic knowledge statistical learning.

- Predictions : Predict y = f(X) where the predictor f is "treated as a black box" and only, matter, the prediction accuracy.
  - reducible error : depend on how well defined is the model
  - irreducible error : error due to intricate variability of the target or effect of missing variable
- Inference : Understanding the relationship between X and y
- Parametric model & non parametric model:
- etc...

### Stanford CS229 Notes

#### Linear Reg

- Batch Gradient Descent : Try to minimize the loss function by reducing the loss on all the training data. It's good for a small dataset but as the dataset get bigger, it become computationally heavy.
- Stochastic Gradient Descent : Try to minimize the loss function by minimizing the loss on each training data. Because, here we approach the problem in an incremental way, it's less heavy on bigger dataset and it's also easy to add new training data without having to retrain the model.

- Parametric model : 
  - have a fixed amount of parameter 
  - we make an assumption on the form of f
- Non parametric model : 
  - the number of parameter grow linearly with the size of the dataset
  - we dont make an assumption on f

### Logistic Regression

- Sigmoid functions
- Newton Method : It's an algorithm to determine x so that for a given function f,  f(x) = 0 (To add)
  
#### Perceptron

- Classification model that is based on a function g that is equal to 0 if < 0 and 1 if > 0
- The optimisation process look like a gradient descent but it is not (the function g is not differentiable)

### Generative Learning Algorithm 


- Discriminative Learning Algorithm (Opposed to Generative Learning Algorithm) based of p(X|y) (given the output, what does X should look like)and prior p(y).
- Gaussian Discrimant Analysis
- Why maximize the joint likelyhood in case of GLA ?