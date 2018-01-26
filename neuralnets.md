# Neural Networks

* A neuron is a logistic unit
* The activation function on a neural network is the sigmoid, which outputs values in the range $(0, 1]$.
* $X$ represents the input (in which you add a bias node $x_0 = 1$).
* $\Theta$ represents the weights
* If a network has $S_j$ units in layer $j$ and $S_{j+1}$ in layer $j+1$, then $\Theta^{(j)}$ will be
of dimension $S_{j+1}\times S_j + 1$.

$a^{(1)} = x$.

$z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$

$a^j = g(z^{(j)})$, where $g$ is the sigmoid function.

$z^{(j+1)} = \Theta^{(j)}a^{(j)}$

$h_{\theta}(x) = a^{(j+1)}g(z^{(j+1)})$
