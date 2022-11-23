# Long Short-term Cognitive Networks

Long Short-term Cognitive Networks (LSTCNs) [1] are a computationally efficient recurrent neural network devoted to time series forecasting. This model supports one-step ahead and multiple-step ahead forecasting for univariate and multivariate time series. However, this model is highly recommended for multivariate settings and multiple-step ahead forecasting. LSTCNs are competitive compared to state-of-the-art recurrent neural networks in terms of forecasting error while being much faster.

## Install

LSTCN can be installed from [PyPI](https://pypi.org/project/lstcn):

<pre>
pip install lstcn
</pre>

## Background

An LSTCN model [1] is a recurrent neural network composed of a collection of Short-term Cognitive Network (STCN) blocks. Each STCN block is a two-layer neural network that implements shallow learning to process an specific time patch. The time patches can be defined as temporal pieces of data resulting from partitioning the time series. Let's assume that $X \in \mathbb{R}^{M \times T}$ is a dataset comprising a multivariate time series. The $k$-th time patch is denoted by the tuple $(X^{(k)}, Y^{(k)})$ where $X^{(k)}, Y^{(k)} \in \mathbb{R}^{C \times (M \times L)}$ where $C$ is the number of instances in a given time patch, $M$ is the number of vairables and $L$ is the number of steps to forecast. Each STCN block passes the knowledge learned in the previous iteration to the next STCN model as prior knowledge defined by a weight matrix:

<p align="center">
  <img src="https://github.com/gnapoles/lstcn/blob/390b3ba84200e0bc58524e87138db4f45ac8f591/figures/LSTCN_diagram.jpg?raw=true" width="800" />
</p>



```math
Y^{(k)}=H^{(k)} W_2^{(k)} \oplus B_2^{(k)}
```

```math
H^{(k)}=f\left(X^{(k)} W_1^{(k)} \oplus B_1^{(k)} \right)
```

```math
\begin{bmatrix} 
W_2^{(k)} \\ 
B_2^{(k)} 
\end{bmatrix} 
= \left( \left( \Phi^{(k)} \right)^{\top} \Phi^{(k)} + \lambda \Omega^{(k)} \right)^{-1} \left( \Phi^{(k)} \right)^{\top} Y^{(k)}
```

where $\Phi^{(k)}=(H^{(k)}|A)$ such that $A_{C \times 1}$ is a column vector filled with ones, $\Omega^{(k)}$ denotes the diagonal matrix of $(\Phi^{(k)})^{\top} \Phi^{(k)}$, while $\lambda \geq 0$ denotes the ridge regularization penalty

## Example Usage



### References
[1] Nápoles, G., Grau, I., Jastrzębska, A., & Salgueiro, Y. (2022). Long short-term cognitive networks. Neural Computing and Applications, 1-13.(https://link.springer.com/article/10.1007/s00521-022-07348-5)

[2] Morales-Hernández, A., Nápoles, G., Jastrzebska, A., Salgueiro, Y., & Vanhoof, K. (2022). Online learning of windmill time series using Long Short-term Cognitive Networks. Expert Systems with Applications, 117721. (https://www.sciencedirect.com/science/article/pii/S0957417422010065)

[3] Grau, I., de Hoop, M., Glaser, A., Nápoles, G., & Dijkman, R. (2022). Semiconductor Demand Forecasting using Long Short-term Cognitive Networks. In Proceedings of the 34th Benelux Conference on Artificial Intelligence and 31st Belgian-Dutch Conference on Machine Learning, BNAIC/BeNeLearn 2022. (https://bnaic2022.uantwerpen.be/wp-content/uploads/BNAICBeNeLearn_2022_submission_4148.pdf)
