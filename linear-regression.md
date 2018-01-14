# Regressão linear

Resume-se ao método dos mínimos quadrados.
A função hipótese é dada por $h_\theta = \theta_0 x + \theta_1$.

A função de custo é definida por $J(\theta_0, \theta_1) = \frac{1}{2m} \sum\limits_{i=1}^m
(h_\theta(x^{(i)}) - y^{(i)})^2$, onde $x^{(i)}$ é o $i$-ésimo exemplo fornecido e $y^{(i)}$
o valor de output correspondente.

## Gradient descent

O _gradient descent_ é um método para encontrar os parâmetros que minimizam a função de custo.
É um algoritmo bem simples que consiste em atualizar, até que sejam convergentes, os valores
de $\theta_j$ $\forall j, 1 \leq j \leq m$, simultaneamente, da seguinte maneira:

$\theta_j -= \alpha\frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1)$, onde $\alpha$ é a
taxa de aprendizado.

## Para múltiplas features

Seja n o número de features. Definiremos $x$ como sendo um vetor de dimensão n contendo as features,
$y$ como o vetor com os outputs esperados, e $\theta$ como sendo o vetor contendo os parâmetros que
minimizam o custo de $h_\theta(x)$. Além disso, definiremos $x_0 = 1$.

Teremos:

$h_\theta(x) = \theta^Tx$

$J(\theta) = \frac{1}{2m} \sum\limits_{i=1}^m (h_\theta(x) - y)^2$

$\theta_j -= \alpha\frac{1}{m}\sum\limits_{i=1}^m (h_\theta(x^{(i)})- y^{(i)})x_j^{(i)}$, onde $x^{(i)}$
são os valores assumidos pelas features do $i$-ésimo exemplo e $x_j^{(i)}$ o valor da $j$-ésima feature.

## Otimizando o _gradient descent_

Para que o algoritmo seja mais rápido, podemos utilizar **feature scaling**, pois se as features
tiverem ordens de grandeza similares, o gradiente converge mais rápido, e **mean normalization**,
pelo mesmo motivo.

Combinando as duas, temos:

$x_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{max(x_j) - min(x_j)}$