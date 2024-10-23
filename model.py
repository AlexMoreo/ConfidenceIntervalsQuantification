import functools
from functools import cache

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.method.base import BaseQuantifier
from scipy.stats import chi2
from sklearn.utils import resample
from abc import ABC, abstractmethod


class ConfidenceRegion:
    def __init__(self, mean, cov, critical):
        self.mean = mean
        self.cov = cov
        self.critical = critical

    @functools.cached_property
    def precision_matrix(self):
        try:
            prec_matrix = np.linalg.inv(self.cov)
        except:
            prec_matrix = None
        return prec_matrix

    def within(self, true_prev):
        if self.precision_matrix is None:
            return False

        # Mahalanobis distance
        diff = true_prev - self.mean
        d_M_squared = np.dot(np.dot(diff.T, self.precision_matrix), diff)  # d_M^2
        within_elipse = d_M_squared <= self.critical
        return within_elipse

class WithCIAbstract(ABC):
    @abstractmethod
    def quantify_ci(self, instances):
        ...

# class WithCI(WithCIAbstract, BaseQuantifier):
#
#     def __init__(self, quantifier:BaseQuantifier, n_samples=100, sample_size=0.5, random_state=None):
#         assert n_samples > 1, f'{n_samples=} must be > 1'
#         assert (type(sample_size)==float and sample_size > 1) or (type(sample_size)==int and sample_size>1), \
#             f'wrong value for {sample_size=}; specify a float (a proportion of the original size) or an integer'
#         self.quantifier = quantifier
#         self.n_samples = n_samples
#         self.sample_size = sample_size
#         self.random_state = random_state
#
#     def fit(self, data: LabelledCollection):
#         return self.quantifier.fit(data)
#
#     def quantify(self, instances):
#         prev_mean, self.prev_cov_last = self.quantify_ci(instances)
#         return prev_mean
#
#     def quantify_ci(self, instances):
#         prevs = []
#         for i in range(self.n_samples):
#             sample_i = resample(instances, n_samples=self.sample_size, random_state=self.random_state)
#             prev_i = self.quantifier.quantify(sample_i)
#             prevs.append(prev_i)
#         prevs = np.asarray(prevs)
#         prev_mean = prevs.mean(axis=0)
#         prev_cov  = np.cov(prevs)
#         return prev_mean, prev_cov


class WithCIAgg(WithCIAbstract, AggregativeQuantifier):

    def __init__(self, quantifier: AggregativeQuantifier, n_samples=100, sample_size=1., confidence_level=0.95, random_state=None, df_red=False):
        assert n_samples > 1, f'{n_samples=} must be > 1'
        assert (type(sample_size) == float and sample_size > 0) or (type(sample_size) == int and sample_size > 1), \
            f'wrong value for {sample_size=}; specify a float (a proportion of the original size) or an integer'
        self.quantifier = quantifier
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.df_red = df_red

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
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

        prevs = np.asarray(prevs)
        mean = prevs.mean(axis=0)
        cov = np.cov(prevs, rowvar=False, ddof=1)

        # critical chi-square value
        n_classes = prevs.shape[1]  # number of variables is the number of classes
        if self.df_red:
            chi2_val = chi2.ppf(confidence_level, df=n_classes-1)  # prevs sum up to 1, so one degrees of freedom less
        else:
            chi2_val = chi2.ppf(confidence_level, df=n_classes)

        return ConfidenceRegion(mean, cov, chi2_val)


    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        return self.quantifier.fit(data, fit_classifier, val_split)

    def quantify_ci(self, instances, confidence_level=None):
        predictions = self.quantifier.classify(instances)
        return self.aggregate_ci(predictions, confidence_level=confidence_level)

    @property
    def classifier(self):
        return self.quantifier.classifier

    def _classifier_method(self):
        return self.quantifier._classifier_method()
