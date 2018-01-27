# Neural Networks

* A neuron is a logistic unit
* The activation function on a neural network is the sigmoid, which outputs values in the range $(0, 1]$.
* $X$ represents the input (in which you add a bias node $x_0 = 1$).
* $\Theta$ represents the weights

## Forward propagation

* If a network has $S_j$ units in layer $j$ and $S_{j+1}$ in layer $j+1$, then $\Theta^{(j)}$ will be
of dimension $S_{j+1}\times S_j + 1$.

$a^{(1)} = x$.

$z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$

$a^j = g(z^{(j)})$, where $g$ is the sigmoid function.

$z^{(j+1)} = \Theta^{(j)}a^{(j)}$

$h_{\theta}(x) = a^{(j+1)}g(z^{(j+1)})$

## Cost function

* training set: $\{(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})\}$.
* $L =$ total number of layers in network.
* $s_l =$ number of units (not counting bias unit) in layer $l$.
* Binary classification: $y = 0$ or $1$, $1$ output unit.
* Multi-class classification: $y \in \mathbb{R}^K$, $K$ output units; $h_\theta(x) \in \mathbb{R}^k$.

$J(\Theta) = - \frac{1}{m} [\sum\limits_{i=1}^m\sum\limits_{k=1}^K y_k^{(i)}log(h_\theta(x^{(i)}))_k + (1 - y_k^{(i)}) log (1 - (h_\theta(x^{(i)}))_k)]+ \frac{\lambda}{2m}\sum\limits_{l=1}^{L-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_l+1}(\Theta_{ji}^{(l)})^2$

## Gradient computation

We need to compute:

* $J(\Theta)$
* $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)$

### Backpropagation

* $\delta_j^{(l)} =$ error of node $j$ in layer $l$.
* For each output unit in a layer $l$, $\delta_j^{(l)} = a_j^{(l)} - y_j = (h_\Theta(x))_j - y_j$
* $\delta^{(l-1)} = (\Theta^{(l-1)})^T\delta^{(l)}.*\:g'(z^{(l-1)})$, where $.*$ is the element-wise multiplication.
* $g'(z^{(l)}) = a^{(l)}.*(1-a^{(l)})$.
* $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) = a_j^{(l)}\delta_i^{(l+1)}$, (ignoring $\lambda$; if $\lambda = 0$).
* It's best to initialize each $\Theta_{ij}^{(l)}$ to a random value in $[-\epsilon, \epsilon]$ than to initialize them all to ones or zeros (as it promotes simmetry between weights): ``Theta1 = rand(10,11)*(2*INIT_EPSILON) - INIT_EPSILON``, the random parameters being the dimension of the Theta matrix.

#### The algorithm:

Set $\Delta_{ij}^{(l)}= 0\:\: \forall l, i, j$.

For $i = 1$ to $m$:

$\qquad1.$ Set $a^{(1)} = x^{(i)}$.

$\qquad2.$ Perform forward propagation to compute $a^{(l)}$ for $l = 2, 3, \ldots, L$.

$\qquad3.$ Using $y^{(i)}$ compute $\delta^{(L)} = a^{(L)}-y^{(i)}$.

$\qquad4.$ Compute $\delta^{(i)}$ for $i = L-1, \ldots, 2$.

$\qquad5.$ $\Delta_{ij}^{(l)} := \Delta_{ij}^{(l)} + a_j^{(l)}\delta_i^{(l+1)}$

Finally, to obtain the partial derivative and update the $\Delta$ matrix:

$$
D_{ij}^{(l)} =  \left\{
\begin{array}{ll}
      \frac{1}{m}\Delta_{ij}^{(l)} + \lambda\Theta_{ij}^{(l)} & if\:j \neq 0 \\
      & \\
       \frac{1}{m}\Delta_{ij}^{(l)} & if\:j = 0 \\
\end{array} \right.
$$

#### Unrolling the matrices

To use functions such as ``fminunc`` we need to have vectors instead of matrices. We can obtain them through:

    thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
    deltaVector = [ D1(:); D2(:); D3(:) ]

And we can roll them back into matrices by:

    Theta1 = reshape(thetaVector(1:110),10,11)
    Theta2 = reshape(thetaVector(111:220),10,11)
    Theta3 = reshape(thetaVector(221:231),1,11)

#### Gradient checking

Backpropagation can be tricky to implement and end up with bugs that hamper performance. We can use gradient checking to verify if it is working correctly.

We approximate the derivative using $\frac{J(\theta + e) - J(\theta - e)}{2e}$. In Octave, we can approximate the partial derivatives using the following code:

    for i = 1:n
        thetaPlus = theta;
        thetaPlus(i) += EPSILON;
        thetaMinus = theta;
        thetaMinus(i) -= EPSILON;
        gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*EPSILON);
    end;

* Implement backpropagation to compute DVec;
* Implement numerical gradient check to compute gradApprox;
* Make sure they give similar values;
* Turn off gradient checking. Use backprop code for learning. This step is very important as if you run numerical gradient computation on every iteration of gradient descent the code will be very slow.

## Training a Neural Network

* Randomly initialize weights.
* Implement forward propagation to get $\h_theta(x^{(i)})$ for any $x^{(i)}$.
* Implement code to compute cost function $J(\Theta)$.
* Implement backprop to compute partial derivatives.
* Use gradient checking to compare partial derivatives computed using backprop vs. using numerical estimate of gradient of $J(\Theta)$ and then disable gradient checking code.
* Use gradient descent of advanced optimization method with backprop to try to minimize $J(\Theta)$ as a function of parameters $\Theta$.
