import functools
from functools import cache, cached_property

import numpy as np
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.method.base import BaseQuantifier
from scipy.stats import chi2, sem, t
from scipy.special import gamma
from sklearn.utils import resample
from abc import ABC, abstractmethod
from scipy.special import softmax, factorial
import copy
from functools import lru_cache



class ConfidenceSimplexAbstract(ABC):

    @abstractmethod
    def mean(self):
        ...

    def ndim(self):
        return len(self.mean())

    @abstractmethod
    def within(self, true_value):
        ...
    
    @lru_cache
    def simplex_portion(self):
        return self.montecarlo_proportion()

    @lru_cache
    def montecarlo_proportion(self, n_trials=10_000):
        uniform_simplex = F.uniform_simplex_sampling(n_classes=self.ndim(), size=n_trials)
        proportion = np.clip(self.within(uniform_simplex), 0., 1.)
        return proportion


def within_ellipse(values, mean, prec_matrix, chi2_critical):
    """
    :param values: a np.ndarray with shape (ndim,) or (n_values,ndim,)
    :param mean: a np.ndarray with the mean of the sample
    :param prec_matrix: a np.ndarray with the precision matrix (inverse of the
        covariance matrix) of the sample. If this inverse cannot be computed
        then None must be passed
    :param chi2_critical: the chi2 critical value

    :return: the fraction of values that are contained in the ellipse
        defined by the mean, the precision matrix, and the chi2_critical.
        If values is only one value, then either 0 (not contained) or 
        1 (contained) is returned.
    """
    if prec_matrix is None:
        return 0.
    
    diff = values - mean  # Mahalanobis distance

    d_M_squared = diff @ prec_matrix @ diff.T  # d_M^2
    if d_M_squared.ndim == 2:
        d_M_squared = np.diag(d_M_squared)

    within_elipse = (d_M_squared <= chi2_critical)

    if isinstance(within_elipse, np.ndarray):
        within_elipse = np.mean(within_elipse)

    return within_elipse*1.0
    

def simplex_volume(n):
    return 1 / factorial(n)


class ConfidenceRegionSimplex(ConfidenceSimplexAbstract):

    def __init__(self, X, confidence_level=0.95, num_checks=100):

        assert 0 < confidence_level < 1, f'{confidence_level=} must be in range(0,1)'

        X = np.asarray(X)

        self.mean_ = X.mean(axis=0)
        self.cov = np.cov(X, rowvar=False, ddof=1)

        try:
            self.precision_matrix = np.linalg.inv(self.cov)
        except:
            self.precision_matrix = None

        self.dim = X.shape[1]
        self.ddof = self.dim - 1

        # critical chi-square value
        self.confidence_level = confidence_level
        self.chi2_critical = chi2.ppf(confidence_level, df=self.ddof)

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)
        small_eigenvalues = self.eigenvalues < 1e-7
        self.eigenvalues[small_eigenvalues]=0
        self.semiaxes = np.sqrt(self.eigenvalues * self.chi2_critical)

        self.num_checks = num_checks

    def mean(self):
        return self.mean_

    def within(self, true_value):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        confidence_level None means that the confidence_level is taken from the __init__
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        return within_ellipse(true_value, self.mean_, self.precision_matrix, self.chi2_critical)
        
    @cached_property
    def _is_within_simplex(self):
        ndim = self.ndim()
        simplex_dim = ndim - 1
        num_checks = self.num_checks
        
        # checks whether all points in the surface of the ellipse are contained in the probability simplex;
        # to do so, generates num_checks points in the surface of a sphere
        points_on_sphere = np.random.normal(size=(num_checks, simplex_dim))
        points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1)[:, np.newaxis]
        # adds missing dimension (all zeros)
        points_on_sphere = np.hstack([np.zeros(shape=(points_on_sphere.shape[0],1)), points_on_sphere])

        # scale by ellipse's axes
        scaled_points = points_on_sphere * self.semiaxes

        # rotate and translate to mean
        ellipsoidal_points = scaled_points.dot(self.eigenvectors.T) + self.mean_

        # check all points in the border satisfy the simplex conditions:
        
        #   1. all non-negative
        all_positive = (ellipsoidal_points>=0).all()
        if not all_positive:
            return False
        
        #   2. must sum to 1
        sum_1 = np.isclose(ellipsoidal_points.sum(axis=1), 1).all()
        if not sum_1:
            return False
        
        return True

    @cached_property
    def _volume(self):
        """
        Calculates the volume of the confidence ellipsoid

        Returns:
        float: The volume of the ellipsoid.
        """
        no_null_semiaxis = self.semiaxes[self.semiaxes>0]
        n = len(no_null_semiaxis)
        
        # Scaling factor for the volume in n dimensions
        volume_factor = (np.pi ** (n / 2)) / gamma(n / 2 + 1)
        
        volume = volume_factor * np.prod(no_null_semiaxis)

        return volume
    
    @lru_cache
    def simplex_portion(self):
        # if self._is_within_simplex:
        #     return self._volume / simplex_volume(self.ndim()-1)
        # else:
        return self.montecarlo_proportion()
            


class ConfidenceRegionCLR(ConfidenceSimplexAbstract):

    def __init__(self, X, confidence_level=0.95, num_checks=100):
        self.clr = CLR()
        Z = self.clr(X)
        self.clean_mean = np.mean(X, axis=0)
        self.conf_region_clr = ConfidenceRegionSimplex(Z, confidence_level=confidence_level, num_checks=0)

    def mean(self):
        #Z_mean = self.conf_region_clr.mean()
        #return self.clr.inverse(Z_mean)
        # the inverse of the CLR does not coincide with the clean mean because the geometric mean
        # requires smoothing the prevalence vectors and this affects the softmax (inverse)
        return self.clean_mean

    def within(self, true_value):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        confidence_level None means that the confidence_level is taken from the __init__
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        transformed_values = self.clr(true_value)
        return self.conf_region_clr.within(transformed_values)
        
    @cached_property
    def _is_within_simplex(self):
        return False
    
    @lru_cache
    def simplex_portion(self):
        return self.montecarlo_proportion()


class ConfidenceIntervals(ConfidenceSimplexAbstract):

    def __init__(self, X, confidence_level=0.95):
        assert 0 < confidence_level < 1, f'{confidence_level=} must be in range(0,1)'

        X = np.asarray(X)

        self.means = X.mean(axis=0)
        alpha = 1-confidence_level
        low_q = (alpha / 2.)*100
        high_q = (1 - alpha / 2.)*100
        I_low, I_high = np.percentile(X, q=[low_q, high_q], axis=0)
        self.conf_intervals_low = I_low
        self.conf_intervals_high = I_high
        # self.sem = sem(X, axis=0, ddof=1)
        # self.nd = X.shape[0]
        #
        # h = t.ppf((1 + confidence_level) / 2., self.nd - 1) * self.sem
        # self.conf_intervals_low = np.clip(self.means-h, a_min=0., a_max=1.)
        # self.conf_intervals_high = np.clip(self.means+h, a_min=0., a_max=1.)
    
    def mean(self):
        return self.means

    def within(self, true_value):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        I_low = self.conf_intervals_low
        I_high = self.conf_intervals_high

        within_intervals = np.logical_and(I_low <= true_value, true_value <= I_high)
        within_all_intervals = np.all(within_intervals, axis=-1, keepdims=True)
        proportion = within_all_intervals.mean()

        return proportion


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
        X = np.asarray(X)
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


