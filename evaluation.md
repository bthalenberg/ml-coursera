# Evaluation

## Evaluate hypothesis

* Split dataset intro training (0.6), cross-validation (0.2) and test (0.2).
* Learn $\Theta$ from training set.
* Pick the best model (i.e. the degree of the polynomial, etc) using the cross-validation set.
* Estimate generalization error on the test set.

To estimate the generalization error we can use the cost function or the misclassification error (percentage of wrongly classified values).

## Determining the problem:

* **High bias (underfitting)**: the cost on the training set is high and on the cross-validation set it is also high.
* **High variance (overfitting)**: the cost on the training set is low and on the cross-validation set it is high.

## How to choose the right model and the right regularization term

1. Create a list of lambdas (i.e. $\lambda\in{0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24}$);
2. Create a set of models with different degrees or any other variants;
3. Iterate through the $\lamda$s and for each one, go through all the models to learn some $\Theta$.
4. Compute the cross validation error using the learned $\theta$ (computed with $\lambda$) on the $J_{cv}(\Theta)$ with $\lambda = 0$.
5. Select the best combo that produces the lowest error on the cross validation set.
6. Using the best combo $\Theta$ and $\lambda$, apply it on $J_{test}(\Theta)$ to see if it has a good generalization of the problem.

## Learning curve

Plot J as a function of m (training set size), artificially reducing m.

If you have a high bias, you'll have the cost on the cross-validation set getting lower until it reaches a straight line, and the cost on the training set getting higher until it reaches a straight line (both lines will have about the same value). If the algorithm is suffering from high bias, getting more training data will not help much.

If you have a high variance, cross-validation and training set will have the same behavior as before (cv will get lower and training will get higher), but the gap between the two lines will be bigger. If the algorithm is suffering from high variance, getting more training data is likely to help.

## Summary

* Get more training examples: fixes high variance
* Try smaller set of features: fixes high variance
* Try getting additional features: fixes high bias
* Try adding polynomial features: fixes high bias
* Try decreasing $\lambda$: fixes high bias
* Try increasing $\lambda$: fixes high variance

* A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
* A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase $\lambda$) to address the overfitting.
* Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

* Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
* Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
