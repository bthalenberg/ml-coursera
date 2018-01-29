# Support Vector Machines

SVMs are basically a logistic regression with a modified cost function in a way that it offers
computational advantages.

$J(\theta) = C \sum\limits_{i=1}^m y^{(i)}\:cost_1(\theta^Tx^{(i)}) + (1-y)\:cost_0(\theta^Tx^{(i)}) + \frac{1}{2}\sum\limits_{i=1}^n \Theta_j^2$,
where $cost_0$ and $cost_1$ are margin cost functions.

They are sometimes called **Large margin classifiers** because the cost is set up in a way that the
classifiers tries to find parameters which make the margins between the two classes the largest as possible.

## Kernels

Way to find a non-linear decision boundary (the use of no kernels is sometimes referred as linear kernel
and yields similar results to logistic regression).

* Pick landmarks (usually the training set values) and, given x, compute a new feature depending on
proximity to the landmark. (For each data point on the training set, compute its similarity to the
other data points and create a feature vector). The similarity $sim(x, l^{(i)})$ is given by $\exp(-\frac{||x - l^{(i)}||^2}{2\sigma^2})$
in the Gaussian kernel, the most common one.
* The new hypothesis is going to be: predict $y = 1$ if $\Theta^Tf \geq 0$.

There is a great choice of kernels beside the linear and the Gaussian ones, such as the chi-squared,
the String, the polynomial -- $(x^Tl + k)^d$, etc. You can use pretty much any function as a kernel,
as long as it satisfies the Mercer's Theorem, so that the SVM runs correctly and does not diverge.

## Picking the parameters

* large C: lower bias, higher variance
* small C: higher bias, lower variance
* large $\sigma^2$: features vary more smoothly; higher bias, lower variance;
* small $\sigma^2$: features vary less smoothly; lower bias, higher variance;

## SVM Packages

Since SVMs are based on many numerical optimizations, it doesn't make sense to code your own. use
a library such as __libsvm, liblinear__.

In these packages, you will need to define the parameters C and $\sigma^2$ and to choose a kernel (and
sometimes provide the similarity function).

If your features need scaling, do that **before** you use the Gaussian kernel.

## Multi-class

You can use a built-in function or apply the one-vs-all technique.

## When to use it

Let $n$ be the number of features and $m$ be the number of training examples.

* If $n \gg m$, use logistic regression or SVM with linear kernel.
* If $m \gg n$, create more features and then use logistic regression or SVM with linear kernel.
* If $n \in [1, 1000]$ and $m \in [10, 10.000]$, use SVM with Gaussian kernel.
