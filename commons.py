import numpy as np
import quapy as qp
from sklearn.linear_model import LogisticRegression
from model import WithCIAgg, BayesianCC
from quapy.method.aggregative import PACC, ACC, CC, PCC, DistributionMatchingY, ExpectationMaximizationQuantifier, EMQ

SAMPLE_SIZE = 500
N_JOBS = -1
N_BAGS_TEST = 100
SEED = 1
USE_PROTOCOL = 'upp'

DATASETS = qp.datasets.UCI_MULTICLASS_DATASETS
BINARY_DATASETS = qp.datasets.UCI_BINARY_DATASETS
BINARY_DATASETS = [d for d in BINARY_DATASETS if d not in ['semeion', 'ctg.1', 'ctg.2', 'ctg.3']]
# BINARY_DATASETS = ['ctg.1', 'ctg.2', 'ctg.3']
MULTICLASS_DATASETS = qp.datasets.UCI_MULTICLASS_DATASETS[::-1]


def newLR():
    return LogisticRegression(max_iter=1000)


QUANTIFIERS = [
        #(CC, 'CC'),
        # (ACC, 'ACC'),
        # (PCC, 'PCC'),
        (PACC, 'PACC'),
        # (DistributionMatchingY, 'DM'),
        # (EMQ, 'SLD')
    ]

METHODS = []
for q, q_name in QUANTIFIERS:
    for method in ['intervals']:#, 'clr', 'region']:
        method_prefix = method[:3].title()
        methods = [
            (f'{q_name}-95-tr1-te500-m{method_prefix}', WithCIAgg(q(newLR()), confidence_level=0.95,  n_train_samples=1, n_test_samples=500, method=method)),
            (f'{q_name}-95-tr500-te1-m{method_prefix}', WithCIAgg(q(newLR()), confidence_level=0.95,  n_train_samples=500, n_test_samples=1, method=method)),
            (f'{q_name}-tr100-te100-m{method_prefix}', WithCIAgg(q(newLR()), confidence_level=0.95,  n_train_samples=100, n_test_samples=100, method=method)),
            (f'{q_name}', q(newLR())),
        ]
        METHODS.extend(methods)


def add_conf_eval_metric(df):
    # adds an evaluation metric for measuring the goodness of a conf region    

    coverage = df['within'].values
    amplitude = df['proportion'].values

    # in this case, we consider a good confidence region one that has good 
    # coverage ('within') and covers a small proportion of the simplex (1-'proportion')
    # and this is implemented in terms of the harmonic mean
    def armonic_mean(a, b):
        eps = 1e-8
        return 2*a*b/(a+b+eps)

    def ratio(coverage, amplitude):
        return coverage/(amplitude+1e-7)
    
    # df['goodness'] = armonic_mean(within, 1-proportion)
    df['goodness'] = np.log(1+ratio(coverage, amplitude))
    
    return df
