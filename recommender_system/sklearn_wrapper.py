import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array


class MC(BaseEstimator):
    """
    An estimator for latent factor collaborative filtering models in Recommender Systems.
    """
    def __init__(self, n_u, n_m, factors=10, n_epochs=250, lmbda=10, gamma=9e-5, solver="sgd"):
        self.n_u = n_u
        self.n_m = n_m
        self.factors = factors
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.gamma = gamma
        self.solver = solver

    def fit(self, X, y):
        """
        Fits all the latent factors for users and items and saves the resulting matrix representations.
        """
        X, y = check_X_y(X, y)

        # Create training matrix
        R = np.zeros((self.n_u, self.n_m))
        for idx, row in enumerate(X):
            R[row[0] - 1, row[1] - 1] = y[idx]

            # Initialize latent factors
        P = 3 * np.random.rand(self.n_u, self.factors)  # Latent factors for users
        Q = 3 * np.random.rand(self.n_m, self.factors)  # Latent factors for movies

        def rmse_score(R, Q, P):
            I = R != 0  # Indicator function which is zero for missing data
            ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
            MSE = ME ** 2
            return np.sqrt(np.sum(MSE) / np.sum(I))  # sum of squared errors

        # Fit with stochastic or batch gradient descent
        train_errors = []
        if self.solver == "sgd":
            # Stochastic GD
            users, items = R.nonzero()
            for epoch in range(self.n_epochs):
                for u, i in zip(users, items):
                    e = R[u, i] - np.dot(P[u, :], Q[i, :].T)  # Error for this observation
                    P[u, :] += self.gamma * (e * Q[i, :] - self.lmbda * P[u, :])  # Update this user's features
                    Q[i, :] += self.gamma * (e * P[u, :] - self.lmbda * Q[i, :])  # Update this movie's features
                train_errors.append(rmse_score(R, Q, P))  # Training RMSE for this pass

        elif self.solver == "batch_gd":
            # Batch GD
            for epoch in range(self.n_epochs):
                ERR = np.multiply(R != 0,
                                  R - np.dot(P, Q.T))  # compute error with present values of Q, P, ZERO if no rating
                P += self.gamma * (np.dot(Q.T, ERR.T).T - self.lmbda * P)  # update rule
                Q += self.gamma * (np.dot(P.T, ERR).T - self.lmbda * Q)  # update rule
                train_errors.append(rmse_score(R, Q, P))  # Training RMSE for this pass
        else:
            print("I'm sorry, we don't recognize that solver.")

        # print("Completed %i epochs, final RMSE = %.2f" %(self.n_epochs, train_errors[-1]))
        self.Q = Q
        self.P = P
        self.train_errors = train_errors

        # Return the estimator
        return self

    def predict(self, X):
        """
        Predicts a vector of ratings from a matrix of user and item ids.
        """
        X = check_array(X)

        y = np.zeros(len(X))
        PRED = np.dot(self.P, self.Q.T)
        for idx, row in enumerate(X):
            y[idx] = PRED[row[0] - 1, row[1] - 1]

        return y

    def score(self, X, y):
        """
        Element-wise root mean squared error.
        """
        yp = self.predict(X)
        err = y - yp
        mse = np.sum(np.multiply(err, err)) / len(err)
        return np.sqrt(mse)
