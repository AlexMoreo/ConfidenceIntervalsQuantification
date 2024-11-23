import functools
from functools import cache, cached_property

import numpy as np
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier, AggregativeCrispQuantifier, _bayesian
from quapy.method.base import BaseQuantifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from confidence import ConfidenceRegionSimplex, ConfidenceRegionCLR, ConfidenceIntervals
from sklearn.utils import resample
from abc import ABC, abstractmethod
import copy
#from quapy.method.aggregative import BayesianCC


class WithCIAbstract(ABC):
    @abstractmethod
    def quantify_ci(self, instances):
        ...


class WithCIAgg(WithCIAbstract, AggregativeQuantifier):

    METHODS = ['intervals', 'region', 'clr']

    def __init__(self,
                 quantifier: AggregativeQuantifier,
                 n_train_samples=1,
                 n_test_samples=100,
                 sample_size=1.,
                 confidence_level=0.95,
                 method='intervals',
                 random_state=None):

        assert isinstance(quantifier, AggregativeQuantifier), \
            f'base quantifier does not seem to be an instance of {AggregativeQuantifier.__name__}'
        assert n_train_samples >= 1, \
            f'{n_train_samples=} must be >= 1'
        assert n_test_samples >= 1, \
            f'{n_test_samples=} must be >= 1'
        assert n_test_samples>1 or n_train_samples>1, \
            f'either {n_test_samples=} or {n_train_samples=} must be >1'
        assert (type(sample_size) == float and sample_size > 0) or (type(sample_size) == int and sample_size > 1), \
            f'wrong value for {sample_size=}; specify a float (a proportion of the original size) or an integer'
        assert method in self.METHODS, \
            f'unknown method; valid ones are {self.METHODS}'

        self.quantifier = quantifier
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.method = method
        self.random_state = random_state

    def _return_conf(self, prevs, confidence_level):
        if self.method == 'intervals':
            return ConfidenceIntervals(prevs, confidence_level=confidence_level)
        elif self.method == 'region':
            return ConfidenceRegionSimplex(prevs, confidence_level=confidence_level)
        elif self.method == 'clr':
            return ConfidenceRegionCLR(prevs, confidence_level=confidence_level)
        else:
            raise ValueError(f'unknown method {self.method}')

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        if type(self.sample_size)==float:
            n_samples = int(len(data) * self.sample_size)
        else:
            n_samples = self.sample_size

        self.quantifiers = []
        if self.n_train_samples==1:
            self.quantifier.aggregation_fit(classif_predictions, data)
            self.quantifiers.append(self.quantifier)
        else:
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

        conf = self._return_conf(prevs, confidence_level)
        prev_estim = conf.mean()

        return prev_estim, conf

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        self.quantifier._check_init_parameters()
        classif_predictions = self.quantifier.classifier_fit_predict(data, fit_classifier, predict_on=val_split)
        self.aggregation_fit(classif_predictions, data)
        return self

    def quantify_ci(self, instances, confidence_level=None):
        predictions = self.quantifier.classify(instances)
        return self.aggregate_ci(predictions, confidence_level=confidence_level)

    @property
    def classifier(self):
        return self.quantifier.classifier

    def _classifier_method(self):
        return self.quantifier._classifier_method()


class BayesianCC(AggregativeCrispQuantifier, WithCIAbstract):
    """
    `Bayesian quantification <https://arxiv.org/abs/2302.09159>`_ method,
    which is a variant of :class:`ACC` that calculates the posterior probability distribution
    over the prevalence vectors, rather than providing a point estimate obtained
    by matrix inversion.

    Can be used to diagnose degeneracy in the predictions visible when the confusion
    matrix has high condition number or to quantify uncertainty around the point estimate.

    This method relies on extra dependencies, which have to be installed via:
    `$ pip install quapy[bayes]`

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: a float in (0, 1) indicating the proportion of the training data to be used,
        as a stratified held-out validation set, for generating classifier predictions.
    :param num_warmup: number of warmup iterations for the MCMC sampler (default 500)
    :param num_samples: number of samples to draw from the posterior (default 1000)
    :param mcmc_seed: random seed for the MCMC sampler (default 0)
    """
    def __init__(self,
                 classifier: BaseEstimator=None,
                 val_split: float = 0.75,
                 num_warmup: int = 500,
                 num_samples: int = 1_000,
                 mcmc_seed: int = 0,
                 confidence_level=0.95):

        super().__init__()

        if num_warmup <= 0:
            raise ValueError(f'parameter {num_warmup=} must be a positive integer')
        if num_samples <= 0:
            raise ValueError(f'parameter {num_samples=} must be a positive integer')

        #if (not isinstance(val_split, float)) or val_split <= 0 or val_split >= 1:
        #    raise ValueError(f'val_split must be a float in (0, 1), got {val_split}')

        #if _bayesian.DEPENDENCIES_INSTALLED is False:
        #    raise ImportError("Auxiliary dependencies are required. Run `$ pip install quapy[bayes]` to install them.")

        #self.classifier = qp._get_classifier(classifier)
        self.classifier = classifier
        
        self.val_split = val_split
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.mcmc_seed = mcmc_seed

        # Array of shape (n_classes, n_predicted_classes,) where entry (y, c) is the number of instances
        # labeled as class y and predicted as class c.
        # By default, this array is set to None and later defined as part of the `aggregation_fit` phase
        self._n_and_c_labeled = None

        # Dictionary with posterior samples, set when `aggregate` is provided.
        self._samples = None

        self.confidence_level = confidence_level

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Estimates the misclassification rates.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the label predictions issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        pred_labels, true_labels = classif_predictions.Xy
        self._n_and_c_labeled = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=self.classifier.classes_)

    def sample_from_posterior(self, classif_predictions):
        if self._n_and_c_labeled is None:
            raise ValueError("aggregation_fit must be called before sample_from_posterior")

        n_c_unlabeled = F.counts_from_labels(classif_predictions, self.classifier.classes_)

        self._samples = _bayesian.sample_posterior(
            n_c_unlabeled=n_c_unlabeled,
            n_y_and_c_labeled=self._n_and_c_labeled,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            seed=self.mcmc_seed,
        )
        return self._samples

    def get_prevalence_samples(self):
        if self._samples is None:
            raise ValueError("sample_from_posterior must be called before get_prevalence_samples")
        return self._samples[_bayesian.P_TEST_Y]

    def get_conditional_probability_samples(self):
        if self._samples is None:
            raise ValueError("sample_from_posterior must be called before get_conditional_probability_samples")
        return self._samples[_bayesian.P_C_COND_Y]

    def aggregate(self, classif_predictions):
        samples = self.sample_from_posterior(classif_predictions)[_bayesian.P_TEST_Y]
        return np.asarray(samples.mean(axis=0), dtype=float)
    
    def quantify_ci(self, instances):
        classif_predictions = self.classify(instances)
        return self.aggregate_ci(classif_predictions)
    
    def aggregate_ci(self, classif_predictions: np.ndarray):
        mean = self.aggregate(classif_predictions)
        prev_samples = self.get_prevalence_samples()
        region = ConfidenceRegionCLR(prev_samples, confidence_level=self.confidence_level)
        return mean, region
