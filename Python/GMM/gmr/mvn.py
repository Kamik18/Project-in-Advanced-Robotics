import numpy as np
from .utils import check_random_state
import scipy as sp
from scipy.stats import chi2, norm
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinvh


def invert_indices(n_features, indices):
    inv = np.ones(n_features, dtype=bool)
    inv[indices] = False
    inv, = np.where(inv)
    return inv

def regression_coefficients(covariance, i_out, i_in, cov_12=None):
    """Compute regression coefficients to predict conditional distribution.

    Parameters
    ----------
    covariance : array, shape (n_features, n_features)
        Covariance of MVN

    i_out : array, shape (n_features_out,)
        Output feature indices

    i_in : array, shape (n_features_in,)
        Input feature indices

    cov_12 : array, shape (n_features_out, n_features_in), optional (default: None)
        Precomputed block of the covariance matrix between input features and
        output features

    Returns
    -------
    regression_coeffs : array, shape (n_features_out, n_features_in)
        Regression coefficients. These can be used to compute the mean of the
        conditional distribution as
        mean[i1] + regression_coeffs.dot((X - mean[i2]).T).T
    """
    if cov_12 is None:
        cov_12 = covariance[np.ix_(i_out, i_in)]
    cov_22 = covariance[np.ix_(i_in, i_in)]
    prec_22 = pinvh(cov_22)
    return cov_12.dot(prec_22)


def condition(mean, covariance, i_out, i_in, X):
    """Compute conditional mean and covariance.

    Parameters
    ----------
    mean : array, shape (n_features,)
        Mean of MVN

    covariance : array, shape (n_features, n_features)
        Covariance of MVN

    i_out : array, shape (n_features_out,)
        Output feature indices

    i_in : array, shape (n_features_in,)
        Input feature indices

    X : array, shape (n_samples, n_features_out)
        Inputs

    Returns
    -------
    mean : array, shape (n_features_out,)
        Mean of the conditional distribution

    covariance : array, shape (n_features_out, n_features_out)
        Covariance of the conditional distribution
    """
    cov_12 = covariance[np.ix_(i_out, i_in)]
    cov_11 = covariance[np.ix_(i_out, i_out)]
    regression_coeffs = regression_coefficients(
        covariance, i_out, i_in, cov_12=cov_12)

    mean = mean[i_out] + regression_coeffs.dot((X - mean[i_in]).T).T
    covariance = cov_11 - regression_coeffs.dot(cov_12.T)
    return mean, covariance

class MVN(object):
    """Multivariate normal distribution.

    Some utility functions for MVNs. See
    http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    for more details.

    Parameters
    ----------
    mean : array-like, shape (n_features), optional
        Mean of the MVN.

    covariance : array-like, shape (n_features, n_features), optional
        Covariance of the MVN.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """

    def __init__(self, mean=None, covariance=None, verbose=0,
                 random_state=None):
        self.mean = mean
        self.covariance = covariance
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.norm = None

        if self.mean is not None:
            self.mean = np.asarray(self.mean)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance)
        
        if self.mean is None:
            raise ValueError("Mean has not been initialized")
        if self.covariance is None:
            raise ValueError("Covariance has not been initialized")

    def sample(self, n_samples):
        """Sample from multivariate normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the MVN.
        """
        return self.random_state.multivariate_normal(
            self.mean, self.covariance, size=(n_samples,))

    def to_norm_factor_and_exponents(self, X):
        """Compute normalization factor and exponents of Gaussian.

        These values can be used to compute the probability density function
        of this Gaussian: p(x) = norm_factor * np.exp(exponents).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        norm_factor : float
            Normalization factor: constant term outside of exponential
            function in probability density function of this Gaussian.

        exponents : array, shape (n_samples,)
            Exponents to compute probability density function.
        """
        X = np.atleast_2d(X)
        n_features = X.shape[1]

        try:
            L = sp.linalg.cholesky(self.covariance, lower=True)
        except np.linalg.LinAlgError:
            # Degenerated covariance, try to add regularization
            L = sp.linalg.cholesky(
                self.covariance + 1e-3 * np.eye(n_features), lower=True)

        X_minus_mean = X - self.mean

        if self.norm is None:
            # Suppress a determinant of 0 to avoid numerical problems
            L_det = max(sp.linalg.det(L), np.finfo(L.dtype).eps)
            self.norm = 0.5 / np.pi ** (0.5 * n_features) / L_det

        # Solve L x = (X - mean)^T for x with triangular L
        # (LL^T = Sigma), that is, x = L^T^-1 (X - mean)^T.
        # We can avoid covariance inversion when computing
        # (X - mean) Sigma^-1 (X - mean)^T  with this trick,
        # since Sigma^-1 = L^T^-1 L^-1.
        X_normalized = sp.linalg.solve_triangular(
            L, X_minus_mean.T, lower=True).T

        exponent = -0.5 * np.sum(X_normalized ** 2, axis=1)

        return self.norm, exponent
    
    def marginalize(self, indices):
        """Marginalize over everything except the given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to keep.

        Returns
        -------
        marginal : MVN
            Marginal MVN distribution.
        """
        return MVN(mean=self.mean[indices],
                   covariance=self.covariance[np.ix_(indices, indices)])
    
    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : MVN
            Conditional MVN distribution p(Y | X=x).
        """
        mean, covariance = condition(
            self.mean, self.covariance,
            invert_indices(self.mean.shape[0], indices), indices, x)
        return MVN(mean=mean, covariance=covariance,
                   random_state=self.random_state)
    
    def to_ellipse(self, factor=1.0):
        """Compute error ellipse.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        angle : float
            Rotation angle of the ellipse.

        width : float
            Width of the ellipse (semi axis, not diameter).

        height : float
            Height of the ellipse (semi axis, not diameter).
        """
        vals, vecs = sp.linalg.eigh(self.covariance[0:2, 0:2])
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(*vecs[:, 0][::-1])  # Inverts the order
        width, height = factor * np.sqrt(vals)
        return angle, width, height
    