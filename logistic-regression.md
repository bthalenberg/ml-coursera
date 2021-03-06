# Regressão logística

Problema de classificação binária. Definimos $h_{\theta}(x) = P(y = 1 | x; \theta)$.

$h_{\theta}(x) = g(\theta^Tx)$, sendo que $g(z) = \frac{1}{1 + e^{-z}}$. Portanto,

$h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}$.

## Fronteira de decisão (_decision boundary_)

Como o output de $h_{\theta}(x)$ é uma probabilidade, precisamos definir um limite
para afirmar que $y = 1$ e classificar os dados. Normalmente, utiliza-se $h_{\theta}x \geq 0.5$
e portanto, $\theta^Tx \geq 0$.

## Função de custo

\[  J(\theta) =  \left\{
\begin{array}{ll}
      - log (h_{\theta}(x)) & se\:y = 1 \\
       - log (1 - h_{\theta}(x)) & se\:y = 0 \\
\end{array}
\right. \]

Equivalentemente,

$J(\theta) = \frac{-1}{m} \sum\limits_{i=1}^m (y^{(i)}\:log(h_{\theta}(x)) + (1-y)\:log (1 - h_{\theta}(x)))$.

## Gradient descent

$\theta_j \leftarrow \alpha\frac{\partial}{\partial\theta_j}J(\theta)$, onde $\alpha$ é a
taxa de aprendizado.

$\theta_j \leftarrow \theta_j - \frac{\alpha}{m}\sum\limits_{i=1}^m [(h_\theta(x^{(i)})- y^{(i)})x_j^{(i)}]$, onde $x^{(i)}$
são os valores assumidos pelas features do $i$-ésimo exemplo e $x_j^{(i)}$ o valor da $j$-ésima feature. Em forma vetorizada,

$\theta \leftarrow \theta - \frac{\alpha}{m}X^T(g(X\theta) - y)$.

## Otimizando

Podemos substituir o _gradient descent_ por outros algoritmos mais eficientes, como o _conjugate gradient_, _BFGS_ e _L-BFGs_. Neles, não precisamos escolher um $\alpha$ manualmente e a convergência é mais rápida.

São computados numericamente, então a recomendação é utilizar uma implementação já pronta.

Exemplo de utilização:

    function [jVal, gradient] = costFunction(theta);

    options = optimset('GradObj', 'on', 'MaxIter', '100');

    initialTheta = zeros(2,1);

    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);


## _One vs. all_

Problema de classificação multiclasse. Consiste em reduzir o problema em uma série de classificações binárias, tratando todas as classes, exceto uma, como a classe negativa, e o resto como a positiva. Repetindo isso para todas as classes, obtemos as diferentes fronteiras de decisão.

## Overfitting

Quando temos muitas features, teremos uma função com grande variância, que se enquadra perfeitamente aos dados de treinamento, mas que falha em generalizar para novos exemplos.

Como resolver?
* Diminuir o número de features (manualmente ou por meio de um algoritmo)
* **Regularização**: Reduzir a magnitude/valores dos parâmetros $\theta_j$.

### Regularização

Função custo regularizada:

$J(\theta) = - [\frac{1}{m} \sum\limits_{i=1}^m (y^{(i)}\:log(h_{\theta}(x)) + (1-y)\:log (1 - h_{\theta}(x)))] + \frac{\lambda}{2m}\sum\limits_{j=1}^n \theta_j^2$, onde $\lambda$ é o parâmetro de regularização.

O algoritmo do gradiente descendente fica:

$\theta_0 \leftarrow \theta_0 - \frac{\alpha}{m}\sum\limits_{i=1}^m [(h_\theta(x^{(i)})- y^{(i)})x_0^{(i)}]$

$\theta_j \leftarrow \theta_j - \alpha [\frac{1}{m}\sum\limits_{i=1}^m [(h_\theta(x^{(i)})- y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j]$

A equação normal fica:

$\theta = (X^TX + \lambda L)^-1XTy$, onde $L$ é uma matriz com 0 no topo esquerdo, 1 no resto da diagonal principal e 0 em todo o resto, e dimensão $(n+1)\times(n+1)$.
