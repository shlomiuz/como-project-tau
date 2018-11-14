import numpy as np
import implicit
from scipy import sparse
from scipy.sparse.linalg import spsolve


class ImplicitALS(object):
    """
    Implicit weighted ALS taken from Hu, Koren, and Volinsky 2008. Designed for alternating least squares and implicit
    feedback based collaborative filtering.
    """
    def __init__(self, lambda_=0.1, alpha=40, epochs=50, factors=20):
        """
        :param lambda_: Used for regularization during alternating least squares. Increasing this value may increase
        bias but decrease variance. Default is 0.1.
        :param alpha: The parameter associated with the confidence matrix discussed in the paper, where
        Cui = 1 + alpha*Rui. The paper found a default of 40 most effective. Decreasing this will decrease the
        variability in confidence between various ratings.
        :param epochs: The number of times to alternate between both user feature vector and item feature vector in
        alternating least squares. More iterations will allow better convergence at the cost of increased computation.
        The authors found 10 iterations was sufficient, but more may be required to converge.
        :param n_factors: The number of latent features in the user/item feature vectors. The paper recommends varying
        this between 20-200. Increasing the number of features may overfit but could reduce bias.
        """
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epochs = epochs
        self.factors = factors

        self.confidence_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_diagonal = None
        self.item_diagonal = None
        self.lambda_diagonal = None

    def build_confidence_matrix(self, training_matrix):
        """
        :param training_matrix: Our matrix of ratings with shape m x n, where m is the number of users and n is the
        number of items. Should be a sparse csr matrix to save space.
        :return:
        """
        self.confidence_matrix = self.alpha * training_matrix

    def build_factor_matrices(self, random_seed):
        """
        :param random_seed: Set the seed for reproducible results
        :return:
        """
        random_state = np.random.RandomState(random_seed)
        self.num_user = self.confidence_matrix.shape[0]
        self.num_item = self.confidence_matrix.shape[1]

        self.user_factors = sparse.csr_matrix(random_state.normal(size=(self.num_user, self.factors)))
        self.item_factors = sparse.csr_matrix(random_state.normal(size=(self.num_item, self.factors)))

    def build_diagonal_matrices(self):
        self.user_diagonal = sparse.eye(self.num_user)
        self.item_diagonal = sparse.eye(self.num_item)
        self.lambda_diagonal = self.lambda_ * sparse.eye(self.factors)

    def train(self):
        """
        :return: The feature vectors for users and items. The dot product of these feature vectors should give you the
        expected "rating" at each point in your original matrix.
        """
        for iter_step in range(self.epochs):
            user_combinations = self.user_factors.T.dot(self.user_factors)
            item_combinations = self.item_factors.T.dot(self.item_factors)

            for u in range(self.num_user):
                conf_samp = self.confidence_matrix[u, :].toarray()
                pref = conf_samp.copy()
                pref[pref != 0] = 1
                CuI = sparse.diags(conf_samp, [0])
                yTCuIY = self.item_diagonal.T.dot(CuI).dot(self.item_diagonal)
                yTCupu = self.item_diagonal.T.dot(CuI + self.item_diagonal).dot(pref.T)
                # Cu - I + I = Cu
                self.user_factors[u] = spsolve(item_combinations + yTCuIY + self.lambda_, yTCupu)

            for i in range(self.num_item):
                conf_samp = self.confidence_matrix[:, i].T.toarray()
                pref = conf_samp.copy()
                pref[pref != 0] = 1
                CiI = sparse.diags(conf_samp, [0])
                xTCiIX = self.user_factors.T.dot(CiI).dot(self.user_factors)
                xTCiPi = self.user_factors.T.dot(CiI + self.user_diagonal).dot(pref.T)
                self.item_factors[i] = spsolve(user_combinations + xTCiIX + self.lambda_diagonal, xTCiPi)

        return self.user_factors, self.item_factors.T

    def train_implicit(self, training_matrix):
        self.user_factors, self.item_factors = implicit\
            .alternating_least_squares((training_matrix * self.alpha).astype('double'),
                                       factors=self.factors,
                                       regularization=self.lambda_,
                                       iterations=self.epochs)

        return self.user_factors, self.item_factors.T
