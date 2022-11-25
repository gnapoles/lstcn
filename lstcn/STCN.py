import numpy as np
from sklearn.base import MultiOutputMixin, BaseEstimator
from sklearn.linear_model import Ridge


class STCN(MultiOutputMixin, BaseEstimator):
    """
    Short-term Cognitive Network

    """

    def __init__(self, W1=None, W2=None, function='hyperbolic', solver='svd', alpha=1e-2):

        """

        Parameters
        ----------
        W1        :  {array-like} of shape (n_features*n_steps+1, n_features*n_steps)
                      Non-learnable weight matrix used as prior knowledge.
        W2        :  {array-like} of shape (n_features*n_steps+1, n_features*n_steps)
                      Weight matrix to be learned from a given time patch.
        function  :  {String} Activation function ('sigmoid', 'hyperbolic')
        solver    :  {String} Regression solver ('svd', 'cholesky', 'lsqr')
        alpha :      {float} Positive penalization for L2-regularization.

        """

        self.W1 = W1
        self.W2 = W2

        self.function = function
        self.alpha = alpha
        self.solver = solver

    def transform(self, X):

        """ Apply tha activation function to the hidden state.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features*n_steps)
            The raw hidden state of the network for transformation.
        Returns
        ----------
        H : {array-like} of shape (n_samples, n_features*n_steps)
            The transformed hidden state of the network.

        """

        if self.function == 'hyperbolic':
            return np.tanh(X)
        elif self.function == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-X))

    def inverse(self, Y):

        """ Compute the inverse of the activation function.

        Parameters
        ----------
        Y : {array-like} of shape (n_samples, n_features*n_steps)
            The values to be transformed with the inverse.
        Returns
        ----------
        X : {array-like} of shape (n_samples, n_features*n_steps)
            The inverse values of Y for the current activation function.
        Issues
        ----------
        Sigmoid    : The values of Y must be in the (0,1) interval.
        Hyperbolic : The values of Y must be in the (-1,1) interval.

        """

        if self.function == 'hyperbolic':
            return np.arctanh(Y)
        elif self.function == 'sigmoid':
            return np.log(Y / (1 - Y))

    def fit(self, X_train, Y_train):

        """ Fit the model to the (X_train, Y_train) data.

        Parameters
        ----------
        X_train : {array-like} of shape (n_samples, n_features*n_steps)
                  The data to be used as the input for the model.
        Y_train : {array-like} of shape (n_samples, n_features*n_steps)
                  The data to be used as the expected output.
        Returns
        ----------
        STCN : Fully trained STCN block ready for usage.

        """

        # combine the input data with the first set of weights
        H = self.transform(np.matmul(self.add_bias(X_train), self.W1))

        # compute W2 and B2 using a regularized regression model
        reg = Ridge(alpha=self.alpha, solver=self.solver, random_state=42).fit(H, self.inverse(Y_train))
        self.W2 = np.transpose(np.c_[reg.coef_, reg.intercept_])

        return self

    def predict(self, X):

        """ Predict the output for the given input.

        Parameters
        ----------
        X :       {array-like} of shape (n_samples, n_features*n_steps)
                  The input data for prediction purposes.
        Returns
        ----------
        Y :       {array-like} of shape (n_samples, n_features*n_steps)
                  The prediction for the input data.

        """

        # combine the input data with the first set of weights
        H = self.transform(np.matmul(self.add_bias(X), self.W1))

        # multiply the current state times the second weight matrix
        return self.transform(np.matmul(self.add_bias(H), self.W2))

    def add_bias(self, X):

        """ Concatenate a bias vector to a given matrix.

        Parameters
        ----------
        X :       {array-like} of shape (n_samples, n_features*n_steps)
                  The matrix to be expanded with the bias vector.
        Returns
        ----------
        Y :       {array-like} of shape (n_samples, n_features*n_steps)
                  The enlarged matrix with the bias vector.

        """

        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
