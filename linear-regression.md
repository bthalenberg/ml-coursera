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

$\theta_j \leftarrow \theta_j - \alpha\frac{1}{m}\sum\limits_{i=1}^m (h_\theta(x^{(i)})- y^{(i)})x_j^{(i)}$, onde $x^{(i)}$
são os valores assumidos pelas features do $i$-ésimo exemplo e $x_j^{(i)}$ o valor da $j$-ésima feature.

## Otimizando o _gradient descent_

Para que o algoritmo seja mais rápido, podemos utilizar **feature scaling**, pois se as features
tiverem ordens de grandeza similares, o gradiente converge mais rápido, e **mean normalization**,
pelo mesmo motivo.

Combinando as duas, temos:

$x_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{max(x_j) - min(x_j)}$

## Regressão polinomial

Basicamente a mesma coisa, é mais uma questão de _feature engineering_.

## Resolvendo analiticamente

Ao invés de resolver por _gradient descent_, podemos resolver analiticamente, resolvendo

$\theta = (X^TX)^{-1}X^Ty$.

Esse método deve ser preferido quando o número de features é pequeno (n $<$ 10.000). Raramente, $X^TX$ não será inversível. Remover _features_ redundantes deve resolver, mas além disso há funções que calculam a pseudo-inversa numericamente.

## Regularização

Função custo regularizada:

$J(\theta) = \frac{1}{2m}[\sum\limits_{i=1}^m (h_{\theta}(x{(i)}) - y^{(i)})^2 + \lambda \sum \limits_{j=1}^n \theta_j^2]$, onde $\lambda$ é o parâmetro de regularização.

O algoritmo do gradiente descendente fica:

$\theta_0 \leftarrow \theta_0 - \frac{\alpha}{m}\sum\limits_{i=1}^m [(h_\theta(x^{(i)})- y^{(i)})x_0^{(i)}]$

$\theta_j \leftarrow \theta_j - \alpha [\frac{1}{m}\sum\limits_{i=1}^m [(h_\theta(x^{(i)})- y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j]$

A equação normal fica:

$\theta = (X^TX + \lambda L)^-1XTy$, onde $L$ é uma matriz com 0 no topo esquerdo, 1 no resto da diagonal principal e 0 em todo o resto, e dimensão $(n+1)\times(n+1)$.
