import functools
from functools import cache, cached_property

import numpy as np
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.method.base import BaseQuantifier
from scipy.stats import chi2
from scipy.special import gamma
from sklearn.utils import resample
from abc import ABC, abstractmethod
from scipy.special import softmax, factorial


class ConfidenceRegion:

    def __init__(self, X, transformation, constraints, confidence_level=0.95):
        assert 0 < confidence_level < 1, f'{confidence_level=} must be in range(0,1)'

        self.transformation = transformation

        X = np.asarray(X)
        Z = transformation(X)

        self.mean_ = X.mean(axis=0)
        self.mean_Z = Z.mean(axis=0)
        self.cov_Z = np.cov(Z, rowvar=False, ddof=1)

        try:
            self.precision_matrix = np.linalg.inv(self.cov_Z)
        except:
            self.precision_matrix = None

        self.xdim = X.shape[1]
        self.zdim = Z.shape[1]
        self.ddof = self.zdim - constraints

        # critical chi-square value
        self.confidence_level = confidence_level
        self.chi2_critical = chi2.ppf(confidence_level, df=self.ddof)

    @functools.lru_cache
    def mean(self):
        return self.mean_

    def within(self, true_value, confidence_level=None):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        confidence_level None means that the confidence_level is taken from the __init__
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        if (confidence_level is None) or (confidence_level==self.confidence_level):
            chi2_critical = self.chi2_critical
        else:
            chi2_critical = chi2.ppf(confidence_level, df=self.ddof)

        if self.precision_matrix is None:
            return False

        true_value = self.transformation(true_value)

        # if X~N(mean, cov) then (X-mean).T cov^(-1) (X-mean) ~ chi2(df)
        diff = true_value - self.mean_Z  # Mahalanobis distance
        # d_M_squared = np.dot(np.dot(diff.T, self.precision_matrix), diff)  # d_M^2
        d_M_squared = diff @ self.precision_matrix @ diff.T  # d_M^2
        if d_M_squared.ndim == 2:
            d_M_squared = np.diag(d_M_squared)

        within_elipse = (d_M_squared <= chi2_critical)

        if isinstance(within_elipse, np.ndarray):
            within_elipse = np.mean(within_elipse)

        return within_elipse

    def volume(self):
        """
        Calculates the volume of a confidence ellipsoid for a given covariance matrix.

        Parameters:
        cov_matrix (numpy.ndarray): The covariance matrix (n x n).
        confidence_level (float): The desired confidence level (e.g., 0.95).

        Returns:
        float: The volume of the ellipsoid.
        """
        def reduce_dimensions(cov, threshold=1e-7):
            # some directions of the ellipse may be 0, we filter those out before computing the volume
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # filter small directions
            valid_idx = eigenvalues > threshold
            filtered_cov = eigenvectors[:, valid_idx] @ np.diag(eigenvalues[valid_idx]) @ eigenvectors[:, valid_idx].T
            return filtered_cov, eigenvalues[valid_idx]

        # Get the eigenvalues of the covariance matrix
        # eigenvalues, _ = np.linalg.eigh(self.cov_Z)

        filtered_cov, eigenvalues = reduce_dimensions(self.cov_Z)
        n = len(eigenvalues)  # Number of dimensions

        # Lengths of the semi-axes
        semi_axes = np.sqrt(eigenvalues * self.chi2_critical)

        # Scaling factor for the volume in n dimensions
        volume_factor = (np.pi ** (n / 2)) / gamma(n / 2 + 1)

        # Calculate the volume of the ellipsoid
        volume = volume_factor * np.prod(semi_axes)

        return volume


def simplex_volume(n):
    return 1 / factorial(n)


def simplex_proportion_covered(conf_region:ConfidenceRegion):
    simplex_dim = conf_region.xdim
    if isinstance(conf_region.transformation, IdentityFunction):
        return conf_region.volume() / simplex_volume(simplex_dim)
    else:
        montecarlo_trials = 10_000
        uniform_simplex = F.uniform_simplex_sampling(n_classes=simplex_dim, size=montecarlo_trials)
        return conf_region.within(uniform_simplex)


class Transformation(ABC):

    @abstractmethod
    def __call__(self, X):
        ...

    @abstractmethod
    def inverse(self, X):
        ...

    def check(self, X, tol=1e-6):
        T = self.__call__(X)
        X_ = self.inverse(T)
        return np.all(np.isclose(X, X_, rtol=tol))


class CLR(Transformation):
    """
    Centered log-ratio
    """
    def __call__(self, X, epsilon=1e-6):
        X = qp.error.smooth(X, epsilon)
        G = np.exp(np.mean(np.log(X), axis=-1, keepdims=True))  # geometric mean
        return np.log(X / G)

    def inverse(self, X):
        return softmax(X, axis=-1)


class IdentityFunction(Transformation):
    # f(X)=X
    def __call__(self, X):
        return X

    def inverse(self, X):
        return X


class WithCIAbstract(ABC):
    @abstractmethod
    def quantify_ci(self, instances):
        ...


class WithCIAgg(WithCIAbstract, AggregativeQuantifier):

    def __init__(self,
                 quantifier: AggregativeQuantifier,
                 n_samples=100,
                 sample_size=1.,
                 confidence_level=0.95,
                 transform=None,
                 random_state=None):

        assert n_samples > 1, f'{n_samples=} must be > 1'
        assert (type(sample_size) == float and sample_size > 0) or (type(sample_size) == int and sample_size > 1), \
            f'wrong value for {sample_size=}; specify a float (a proportion of the original size) or an integer'
        assert transform in [None, 'clr'], 'unknown transformation'
        self.quantifier = quantifier
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.transform = transform
        self.random_state = random_state

    def _set_transformation(self):
        if self.transform == 'clr':
            self.transform_fn = CLR()
            self.constraints = 0
        elif self.transform is None:
            self.transform_fn = IdentityFunction()
            self.constraints = 1
        else:
            raise NotImplementedError(f'transformation {self.transform} not supported')

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self._set_transformation()
        return self.quantifier.aggregation_fit(classif_predictions, data)

    def aggregate(self, classif_predictions: np.ndarray):
        prev_mean, self.prev_cov_last = self.aggregate_ci(classif_predictions)
        return prev_mean

    def aggregate_ci(self, classif_predictions: np.ndarray, confidence_level=None):
        if confidence_level is None:
            confidence_level = self.confidence_level

        if type(self.sample_size)==float:
            n_samples = int(classif_predictions.shape[0] * self.sample_size)
        else:
            n_samples = self.sample_size

        prevs = []
        for i in range(self.n_samples):
            sample_i = resample(classif_predictions, n_samples=n_samples, random_state=self.random_state)
            prev_i = self.quantifier.aggregate(sample_i)
            prevs.append(prev_i)

        conf_region = ConfidenceRegion(prevs, self.transform_fn, constraints=self.constraints, confidence_level=confidence_level)
        prev_estim = conf_region.mean()

        return prev_estim, conf_region

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        self._set_transformation()
        return self.quantifier.fit(data, fit_classifier, val_split)

    def quantify_ci(self, instances, confidence_level=None):
        predictions = self.quantifier.classify(instances)
        return self.aggregate_ci(predictions, confidence_level=confidence_level)

    @property
    def classifier(self):
        return self.quantifier.classifier

    def _classifier_method(self):
        return self.quantifier._classifier_method()
