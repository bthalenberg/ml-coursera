# Unsupervised learning

* data that doesn't have any labels already associated with it
* tries to find structures/patterns
    * clustering

## Clustering: k-means

* randomly initializes k points (cluster centroids), where k is the number of clusters desired
* finds the points which are closest to each centroid (cluster assignment)
* repeats until convergence:
    * moves the centroid to the average of each cluster
    * redoes the cluster assignment step

The **optimization objective** is to minimize the cost function

$$ J(c^{(1)}, \ldots, c^{(m)}, \mu_1, \ldots, \mu_k) = \frac{1}{m} \sum\limits^m_{i = 1} ||x^{(i)} - \mu_{c_{(i)}}||^2 $$

It is important to randomly initialize the cluster centroids to multiple times to avoid local optima.
To do that, randomly pick k training examples and set $\mu_1, \ldots, \mu_k$ equal to these examples.

The algorithm is as follows:

    for i = 1 to 100:
        randomly initialize k-means
        run k-means
        compute cost functions
    pick clustering with lowest cost

Random initializations are only necessary when dealing with small k ($2 \leq k \leq 10$).

To chose the value of k you can try using the **elbow method**: compute J for different k and plot it against its cost. Pick the k at the elbow of the curve. Sometimes, however, there is no clear elbow.

## Dimensionality reduction: PCA

* data compression
* visualization
* speed up learning algorithm

It is a technique which uses an orthogonal transformation to reduce components keeping the largest possible variance.

* computes the covariance matrix of the data $C = \frac{1}{m} X^TX$
* uses the SVD function ``[U, S, V] = svd(C)`` to compute the eigenvectors.
* The principal components, $U_{reduce}$ are the first k columns of U.
* $z =  U_{reduce}x$

Before using it, it's important
to normalize the data. The top k components are the first k columns of U.

To recover an approximation of the original data from the compressed one you can do $z = U^T_{reduce}x$.

To choose k you can verify the percentage of the variance retained by

$$\frac{\sum\limits^k_{i=1}S_{ii}}{\sum\limits^m_{i=1}S_{ii}}
