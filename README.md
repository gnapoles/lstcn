# Long Short-term Cognitive Networks

The Long Short-term Cognitive Network (LSTCN) model [1] is an efficient recurrent neural network for time series forecasting. It supports both one-step-ahead and multiple-step-ahead forecasting of univariate and multivariate time series. The LSTCN model is competitive compared to state-of-the-art recurrent neural networks such as LSTM and GRU in terms of forecasting error while being much faster.

## Install

LSTCN can be installed from [PyPI](https://pypi.org/project/lstcn)

<pre>
pip install lstcn
</pre>

## Background

An LSTCN model [1] can be defined as a recurrent neural network composed of sequentially-ordered Short-term Cognitive Network (STCN) blocks [2]. Each STCN block is a two-layer neural network that uses shallow learning to fit the model to a specific time patch, which is then transfered to the following block. Time patches are temporal pieces of data resulting from partitioning the time series. 

Let us assume that $X \in \mathbb{R}^{M \times T}$ is a multivariate time series. The $k$-th time patch is denoted by the tuple $(X^{(k)}, Y^{(k)})$ where $X^{(k)}, Y^{(k)} \in \mathbb{R}^{C \times (M \times L)}$ where $C$ is the number of instances, $M$ is the number of features and $L$ is the number of steps to be forecast. Each STCN block passes the knowledge learned in the previous iteration $(W_2^{(k)}$ and $B_2^{(k)})$ to the next STCN model as prior knowledge to perform reasoning.

<p align="center">
  <img src="https://github.com/gnapoles/lstcn/blob/main/figures/LSTCN_diagram.jpg?raw=true" width="800" />
</p>

The input gate operates with the prior knowledge matrix $W_1^{(k)} \in \mathbb{R}^{N \times N}$ with $X^{(k)} \in \mathbb{R}^{N \times N}$ and the prior bias matrix $B_1^{(k)} \in \mathbb{R}^{1 \times N}$ such that $N=(M \times L)$. Prior matrices $W_1^{(k)}$ and $B_1^{(k)}$ are transferred from the previous STCN block and remain locked during the current learning phase. The input gate outputs a temporal state $H^{(k)} \in \mathbb{R}^{C \times N}$ with the knowledge that the block would have produced for $X^{(k)}$ if no further learning would have been performed to obtain $Y^{(k)}$. Such an adaptation is done in the output gate where the temporal state is operated with the learnable weight matrices $W_2^{(k)} \in \mathbb{R}^{N \times N}$ and $B_2^{(k)} \in \mathbb{R}^{1 \times N}$.

```math
H^{(k)}=f\left(X^{(k)} W_1^{(k)} \oplus B_1^{(k)} \right)
```

```math
\hat{Y}^{(k)}=f\left(H^{(k)} W_2^{(k)} \oplus B_2^{(k)} \right)
```

where $\hat{Y}^{(k)}$ denotes the predicted output, while $\oplus$ performs a matrix-vector addition by operating each row of a given matrix with a vector. Similar to other gated recurrent neural networks, the learning process takes place inside each STCN block. Given a temporal state $H^{(k)}$ and the block's expected output $Y^{(k)}$, we can compute the matrices $W_2^{(k)} \in \mathbb{R}^{N \times N}$ and $B_2^{(k)} \in \mathbb{R}^{1 \times N}$ using the following deterministic rule:

```math
\begin{bmatrix} 
W_2^{(k)} \\ 
B_2^{(k)} 
\end{bmatrix} 
= \left( \left( \Phi^{(k)} \right)^{\top} \Phi^{(k)} + \lambda \Omega^{(k)} \right)^{-1} \left( \Phi^{(k)} \right)^{\top} f^{-1}(Y^{(k)})
```

where $\Phi^{(k)}=(H^{(k)}|A)$ such that $A_{C \times 1}$ denotes a column vector filled with ones, $\Omega^{(k)}$ denotes the diagonal matrix of $(\Phi^{(k)})^{\top} \Phi^{(k)}$, while $\lambda \geq 0$ denotes the ridge regularization penalty. This deterministic learning rule assumes that the neuron's activation values in the inner layer are standardized.

## Example Usage

The syntax for the usage of LSTCN is compatible with scikit-learn library.

### Training

First create an LSTCN object specifying the number of features and the number of steps to predict ahead:

```python
model = LSTCN(n_features, n_steps)
```

Optionally, you can also specify the number of STCN blocks in the network (`n_blocks`), the activation function (`function`), the regression solver (`solver`) and the regularization penalization parameter (`alpha`). For more details check the documentation of the LSTCN class.

For training a LSTCN model simply call the fit method:

```python
model.fit(X_train,Y_train)
```

### Hyperparameter tuning

Use walk forward cross-validation and grid search (or any other suitable validation strategy from scikit-learn) for selecting the best-performing model:

```python
tscv = TimeSeriesSplit(n_splits=5)
scorer = make_scorer(model.score, greater_is_better=False)
param_search = {
    'alpha': [1.0E-3, 1.0E-2, 1.0E-1],
    'n_blocks': range(2, 6)
}

gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, refit=True,
                       n_jobs=-1, error_score='raise', scoring=scorer)
gsearch.fit(X_train, Y_train)
best_model = gsearch.best_estimator_
```

### Prediction

For predicting new data use the method predict:

```python
Y_pred = best_model.predict(X_test)
```

For example, this is the prediction for the target series `oil temperature` of the `ETTh1` dataset [5] containing 17420 records of electricity transformer temperatures in China:

<p align="center">
  <img src="https://github.com/gnapoles/lstcn/blob/main/figures/example_pred.png?raw=true" width="1400" />
</p>

In this example, we used 80% of the dataset for training and validation, and 20% for testing. The mean absolute error for the training set is 0.0355, while the test error is 0.0192. More importantly, the model's hyperparameter tuning (exploring 15 models) runs in 3.8599 seconds!

A jupyter notebook with all details is available in the [examples](https://github.com/gnapoles/lstcn/blob/main/examples) folder.

## References

If you use the LSTCN model in your research please cite the following papers:

[1] Nápoles, G., Grau, I., Jastrzębska, A., & Salgueiro, Y. (2022). Long short-term cognitive networks. Neural Computing and Applications, 1-13. [paper](https://link.springer.com/article/10.1007/s00521-022-07348-5) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:tsqxxO4Ul0kJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaCl0s:AAGBfm0AAAAAY32Ej0sEhR2wzKa7dk6C4kVxUT3em6HS&scisig=AAGBfm0AAAAAY32Ej-1zPkScA5cUw8kSxfjYNDERIFe1&scisf=4&ct=citation&cd=-1&hl=en)

[2] Nápoles, G., Vanhoenshoven, F., & Vanhoof, K. (2019). Short-term cognitive networks, flexible reasoning and nonsynaptic learning. Neural Networks, 115, 72-81. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608019300930) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:WE6oovxx-9gJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaDEbk:AAGBfm0AAAAAY32FCbnEY_3UOTzV4qh2Jkjw8uWRKmkg&scisig=AAGBfm0AAAAAY32FCXb7V_h61rxVMwqW-tIpnRav5ps2&scisf=4&ct=citation&cd=-1&hl=en)

Some application papers with nice examples and further explanations:

[3] Morales-Hernández, A., Nápoles, G., Jastrzebska, A., Salgueiro, Y., & Vanhoof, K. (2022). Online learning of windmill time series using Long Short-term Cognitive Networks. Expert Systems with Applications, 117721. [paper](https://www.sciencedirect.com/science/article/pii/S0957417422010065) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:zw7eSIZeni8J:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaDLaY:AAGBfm0AAAAAY32FNaZ4Y4UCT9Pi0MyrcnXkbVr9ZKQK&scisig=AAGBfm0AAAAAY32FNS8iIT36tfp463gOvpckF52eUHpt&scisf=4&ct=citation&cd=-1&hl=en)

[4] Grau, I., de Hoop, M., Glaser, A., Nápoles, G., & Dijkman, R. (2022). Semiconductor Demand Forecasting using Long Short-term Cognitive Networks. In Proceedings of the 34th Benelux Conference on Artificial Intelligence and 31st Belgian-Dutch Conference on Machine Learning, BNAIC/BeNeLearn 2022. [paper](https://bnaic2022.uantwerpen.be/wp-content/uploads/BNAICBeNeLearn_2022_submission_4148.pdf) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:d8vQmLWkfxoJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEOqYxeaDRPY:AAGBfm0AAAAAY32FXPaTi5GsMnukoQWrf0Om83a-J6W6&scisig=AAGBfm0AAAAAY32FXC9uZn6HZlt2vf6hQPhocM_e53y2&scisf=4&ct=citation&cd=-1&hl=en)

Dataset used in the example:

[5] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 11106-11115). [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17325) [bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:VigqltkXN1QJ:scholar.google.com/&output=citation&scisdr=CgXfrbsrEPaI6B_rEX0:AAGBfm0AAAAAY4PtCX1Nw5q5QeLUSLGippUjAq9GB4dc&scisig=AAGBfm0AAAAAY4PtCStnIoFzxpsoHCwvNsGMCy_qkZUZ&scisf=4&ct=citation&cd=-1&hl=en)
