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

        within_elipse = False
        if self.precision_matrix is not None:
            true_value = self.transformation(true_value)

            # if X~N(mean, cov) then (X-mean).T cov^(-1) (X-mean) ~ chi2(df)
            diff = true_value - self.mean_Z  # Mahalanobis distance

            d_M_squared = diff @ self.precision_matrix @ diff.T  # d_M^2
            if d_M_squared.ndim == 2:
                d_M_squared = np.diag(d_M_squared)

            within_elipse = (d_M_squared <= chi2_critical)

            if isinstance(within_elipse, np.ndarray):
                within_elipse = np.mean(within_elipse)

        return within_elipse*1.0
    
    @cache
    def reduced_cov(self):
        threshold=1e-7
        # if this ellipse has been computed from points on a simplex, then one direction is 0, 
        # and this dimesion needs be filtered out before computing the volume
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_Z)
        # filter small directions
        valid_idx = eigenvalues > threshold
        filtered_cov = eigenvectors[:, valid_idx] @ np.diag(eigenvalues[valid_idx]) @ eigenvectors[:, valid_idx].T
        return filtered_cov, eigenvalues[valid_idx], eigenvectors[valid_idx]

    
    @cached_property
    def semiaxes(self):        
        # Get the eigenvalues of the covariance matrix
        _, eigenvalues, _ = self.reduced_cov()

        # Lengths of the semi-axes
        self.semiaxes_ = np.sqrt(eigenvalues * self.chi2_critical)
        return self.semiaxes_
    
    @cached_property
    def dimensions(self):
        return len(self.semiaxes)
    
    def is_within_simplex(self, simplex_dim):
        if self.dimensions != simplex_dim:
            return False
        
        # checks whether all points in the surface of the ellipse are contained in the probability simplex
        num_samples=100

        # generates num_samples points in the surface of a sphere
        points_on_sphere = np.random.normal(size=(num_samples, len(self.mean_Z)))
        points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1)[:, np.newaxis]         
        
        # scale by ellipse's axes
        _, eigenvalues, eigenvectors = self.reduced_cov()
        scaled_points = points_on_sphere * self.semiaxes  
        
        # rotate
        ellipsoidal_points = scaled_points.dot(eigenvectors.T)  # Rotar los puntos
        
        # Paso 6: Trasladar al centro definido por mu
        ellipsoidal_points += self.mean_Z[:len(eigenvalues)]  # Mover al centro de la elipsoide, solo las dimensiones válidas
        
        return ellipsoidal_points


    def volume(self):
        """
        Calculates the volume of a confidence ellipsoid for a given covariance matrix.

        Returns:
        float: The volume of the ellipsoid.
        """
        semiaxes = self.semiaxes
        n = self.dimensions
        
        # Scaling factor for the volume in n dimensions
        volume_factor = (np.pi ** (n / 2)) / gamma(n / 2 + 1)

        volume = volume_factor * np.prod(semiaxes)

        return volume


class ConfidenceIntervals:

    def __init__(self, X, confidence_level=0.95):
        assert 0 < confidence_level < 1, f'{confidence_level=} must be in range(0,1)'

        X = np.asarray(X)

        self.means = X.mean(axis=0)
        self.sem = sem(X, axis=0, ddof=1)
        self.nd = X.shape[1]

        h = t.ppf((1 + confidence_level) / 2., self.nd - 1) * self.sem
        self.conf_intervals_low = np.clip(self.means-h, a_min=0., a_max=1.)
        self.conf_intervals_high = np.clip(self.means+h, a_min=0., a_max=1.)

    def mean(self):
        return self.means

    def within(self, true_value, confidence_level=None):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        confidence_level None means that the confidence_level is taken from the __init__
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        if (confidence_level is None) or (confidence_level==self.confidence_level):
            I_low = self.conf_intervals_low
            I_high = self.conf_intervals_high
        else:
            h = t.ppf((1 + confidence_level) / 2., self.nd - 1) * self.sem
            I_low = np.clip(self.means - h, a_min=0., a_max=1.)
            I_high = np.clip(self.means + h, a_min=0., a_max=1.)

        within_intervals = np.logical_and(I_low < true_value, true_value < I_high)
        within_all_intervals = np.all(within_intervals, axis=-1, keepdims=True)
        proportion = within_all_intervals.mean()

        return proportion

    def volume(self):
        """
        Calculates the volume of the cube

        Returns:
        float: The volume of the cube
        """
        amp = self.conf_intervals_high - self.conf_intervals_low
        volume = np.prod(amp)
        return float(volume)


def simplex_volume(n):
    return 1 / factorial(n)


def simplex_proportion_covered(conf:[ConfidenceRegion,ConfidenceIntervals], simplex_dim):
    n_classes = simplex_dim+1
    def monte_carlo_approx(montecarlo_trials=10_000):        
        uniform_simplex = F.uniform_simplex_sampling(n_classes=n_classes, size=montecarlo_trials)
        proportion = conf.within(uniform_simplex)
        return proportion

    if isinstance(conf, ConfidenceRegion) and not isinstance(conf.transformation, IdentityFunction):
        raise NotImplemented()
    else:
        proportion = conf.volume() / simplex_volume(simplex_dim)
    return min(proportion, 1.)


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

    METHODS = ['intervals', 'region']

    def __init__(self,
                 quantifier: AggregativeQuantifier,
                 n_train_samples=1,
                 n_test_samples=100,
                 sample_size=1.,
                 confidence_level=0.95,
                 transform=None,
                 method='intervals',
                 random_state=None):

        assert n_train_samples >= 1, f'{n_train_samples=} must be >= 1'
        assert n_test_samples >= 1, f'{n_test_samples=} must be >= 1'
        assert n_test_samples>1 or n_train_samples>1, f'either {n_test_samples=} or {n_train_samples=} must be >1'
        assert (type(sample_size) == float and sample_size > 0) or (type(sample_size) == int and sample_size > 1), \
            f'wrong value for {sample_size=}; specify a float (a proportion of the original size) or an integer'
        assert transform in [None, 'clr'], 'unknown transformation'
        assert method in self.METHODS, f'unknown method; valid ones are {self.METHODS}'
        assert transform is None or method == 'region', \
            (f'invalid combination of {method=} and {transform=}; '
             f'transformations can only be coupled with "region"')
        self.quantifier = quantifier
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.transform = transform
        self.method = method
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

        if type(self.sample_size)==float:
            n_samples = int(len(classif_predictions) * self.sample_size)
        else:
            n_samples = self.sample_size

        self.quantifiers = []
        full_index = np.arange(len(data))
        with qp.util.temp_seed(self.random_state):
            for i in range(self.n_train_samples):
                quantifier = copy.deepcopy(self.quantifier)
                index = resample(full_index, n_samples=n_samples)
                classif_predictions_i = classif_predictions.sampling_from_index(index)
                data_i = data.sampling_from_index(index)
                quantifier.aggregation_fit(classif_predictions_i, data_i)
                self.quantifiers.append(quantifier)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        prev_mean, self.confidence = self.aggregate_ci(classif_predictions)
        return prev_mean

    def aggregate_ci(self, classif_predictions: np.ndarray, confidence_level=None):
        if confidence_level is None:
            confidence_level = self.confidence_level

        if type(self.sample_size)==float:
            n_samples = int(classif_predictions.shape[0] * self.sample_size)
        else:
            n_samples = self.sample_size

        prevs = []
        with qp.util.temp_seed(self.random_state):
            for quantifier in self.quantifiers:
                for i in range(self.n_test_samples):
                    sample_i = resample(classif_predictions, n_samples=n_samples)
                    prev_i = quantifier.aggregate(sample_i)
                    prevs.append(prev_i)

        if self.method == 'region':
            conf = ConfidenceRegion(prevs, self.transform_fn, constraints=self.constraints, confidence_level=confidence_level)
        elif self.method == 'intervals':
            conf = ConfidenceIntervals(prevs, confidence_level=confidence_level)

        prev_estim = conf.mean()

        return prev_estim, conf

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        self._set_transformation()
        self.quantifier._check_init_parameters()
        classif_predictions = self.quantifier.classifier_fit_predict(data, fit_classifier, predict_on=val_split)
        return self.aggregation_fit(classif_predictions, data)

    def quantify_ci(self, instances, confidence_level=None):
        predictions = self.quantifier.classify(instances)
        return self.aggregate_ci(predictions, confidence_level=confidence_level)

    @property
    def classifier(self):
        return self.quantifier.classifier

    def _classifier_method(self):
        return self.quantifier._classifier_method()
