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

The algorithm:

Set $\Delta_{ij}^{(l)}= 0\:\: \forall l, i, j$.

For $i = 1$ to $m$:

$\qquad$ Set $a^{(1)} = x^{(i)}$.

$\qquad$ Perform forward propagation to compute $a^{(l)}$ for $l = 2, 3, \ldots, L$.

$\qquad$ Using $y^{(i)}$ compute $\delta^{(L)} = a^{(L)}-y^{(i)}$.

$\qquad$ Compute $\delta^{(i)}$ for $i = L-1, \ldots, 2$.

$\qquad$ $\Delta_{ij}^{(l)} := \Delta_{ij}^{(l)} + a_j^{(l)}\delta_i^{(l+1)}$.

\[
D_{ij}^{(l)} =  \left\{
\begin{array}{ll}
      \frac{1}{m}\Delta_{ij}^{(l)} + \lambda\Theta_{ij}^{(l)} & if\:j \neq 0 \\
       \frac{1}{m}\Delta_{ij}^{(l)} & if\:j = 0 \\
\end{array}
\right.
\]
