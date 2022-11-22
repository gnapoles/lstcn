import numpy as np
from sklearn.base import MultiOutputMixin, BaseEstimator
from sklearn.metrics import mean_absolute_error
from lstcn.STCN import STCN


class LSTCN(MultiOutputMixin, BaseEstimator):
    """
    Long Short-term Cognitive Network

    """

    def __init__(self, n_features,
                 n_steps, n_blocks=5, function='clip', solver='svd', alpha=1e-2, ):

        """

        Parameters
        ----------
        n_features    :  {int} Number of features in the time series.
        n_steps       :  {int} Number of steps-ahead to be forecast.
        n_blocks      :  {int} Number of STCN blocks in the network.
        function      :  {String} Activation function ('sigmoid', 'tanh', 'clip')
        solver        :  {String} Regression solver ('svd', 'cholesky', 'lsqr')
        alpha         :  {float} Positive penalization for regularization.

        """

        self.stcn = None
        self.n_steps = n_steps
        self.n_blocks = n_blocks
        self.n_features = n_features

        self.function = function
        self.solver = solver
        self.alpha = alpha

    def fit(self, X_train, Y_train):

        """ Fit the model to the (X_train, Y_train) data

        Parameters
        ----------
        X_train : {array-like} of shape (n_samples, n_features*n_steps)
                  The data to be used as the input for the model.
        Y_train : {array-like} of shape (n_samples, n_features*n_steps)
                  The data to be used as the expected output.
        Returns
        ----------
        LSTCN : Fully trained LSTCN model ready for usage.

        """

        errors = []

        # the number of instances must be divisible by n_blocks
        if X_train.shape[0] % self.n_blocks != 0:
            multiplier = int(X_train.shape[0] / self.n_blocks)
            X_train = X_train[-(multiplier * self.n_blocks):, :]
            Y_train = Y_train[-(multiplier * self.n_blocks):, :]

        # number of instances in each iteration
        m_batch = int(X_train.shape[0] / self.n_blocks)

        # initialize initial state randomly
        np.random.seed(42)
        init_matrix = -0.5 + np.random.rand(
            self.n_steps * self.n_features + 1, self.n_steps * self.n_features)

        # loop through the input batch by batch
        for i in range(0, X_train.shape[0], m_batch):
            X_train_mini = X_train[i:i + m_batch]
            Y_train_mini = Y_train[i:i + m_batch]

            # build the STCN for the current time patch
            model = STCN(W1=init_matrix, function=self.function,
                         solver=self.solver, alpha=self.alpha)

            # fit the model to the mini-batch
            model.fit(X_train_mini, Y_train_mini)
            init_matrix = np.tanh(model.W2)

        # save the last STCN at the end of training
        self.stcn = model
        return self

    def predict(self, X_test):

        """ Predict the output for the given input

        Parameters
        ----------
        X :       {array-like} of shape (n_samples, n_features*n_steps)
                  The input data for prediction purposes.
        Returns
        ----------
        Y :       {array-like} of shape (n_samples, n_features*n_steps)
                  The prediction for the input data.

        """
        return self.stcn.predict(X_test)

    def score(self, Y_pred, Y_true):

        """ Compute the mean absolute error between Y_pred and Y_true

        Parameters
        ----------
        Y_pred :  {array-like} of shape (n_samples, n_features*n_steps)
                  The output predicted by the LSTCN model.
        Y_true :  {array-like} of shape (n_samples, n_features*n_steps)
                  The ground-truth output.
        Returns
        ----------
        The mean absolute error as the score.

        """
        return mean_absolute_error(Y_pred, Y_true)