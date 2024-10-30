import quapy as qp
from sklearn.linear_model import LogisticRegression
from model import WithCIAgg
from quapy.method.aggregative import PACC


SAMPLE_SIZE = 500
N_JOBS = -1
N_BAGS_TEST = 50
N_BAGS_TEST = 100
SEED = 1
USE_PROTOCOL = 'upp'

DATASETS = qp.datasets.UCI_MULTICLASS_DATASETS
BINARY_DATASETS = qp.datasets.UCI_BINARY_DATASETS
BINARY_DATASETS = [d for d in BINARY_DATASETS if d not in ['semeion', 'ctg.1', 'ctg.2', 'ctg.3']]
# BINARY_DATASETS = ['ctg.1', 'ctg.2', 'ctg.3']
MULTICLASS_DATASETS = qp.datasets.UCI_MULTICLASS_DATASETS


def newLR():
    return LogisticRegression(max_iter=3000)


METHODS = [
    # ('PACC-99', WithCIAgg(PACC(newLR()), confidence_level=0.99), {}), # wrap_hyper(logreg_grid)),
    # ('PACC-95-0.5-I', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=0.5, method='region'), {}), # wrap_hyper(logreg_grid)),
    # ('PACC-95-0.5-I-int', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=0.5, method='intervals'), {}),  # wrap_hyper(logreg_grid)),
    # ('PACC-95-1-I-int', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., method='intervals'), {}),  # wrap_hyper(logreg_grid)),
    # ('PACC-95-0.5-CLR', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=0.5, transform='clr', method='region'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-tr1-te500-s1-fI-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=1, n_test_samples=500, method='region'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-tr500-te1-s1-fI-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=500, n_test_samples=1, method='region'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-tr100-te100-s1-fI-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=100, n_test_samples=100, method='region'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-tr1-te500-s1-fCLR-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=1, n_test_samples=500, method='region', transform='clr'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-tr500-te1-s1-fCLR-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=500, n_test_samples=1, method='region', transform='clr'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-tr100-te100-s1-fCLR-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=100, n_test_samples=100, method='region', transform='clr'), {}), # wrap_hyper(logreg_grid)),
    # ('PACC-95-tr1-te100-s1-fCLR-mReg', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., n_train_samples=1, n_test_samples=100, transform='clr', method='region'), {}), # wrap_hyper(logreg_grid)),
    # ('PACC-95-1-ddof-1-optim', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., df_red=True), wrap_hyper(logreg_grid)), # wrap_hyper(logreg_grid)),
    # ('PACC-90', WithCIAgg(PACC(newLR()), confidence_level=0.90), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-99', WithCIAgg(EMQ(newLR()), confidence_level=0.99), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-95', WithCIAgg(EMQ(newLR()), confidence_level=0.95, sample_size=0.5), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-95-1', WithCIAgg(EMQ(newLR()), confidence_level=0.95, sample_size=1.), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-95-1-ddof-1', WithCIAgg(EMQ(newLR()), confidence_level=0.95, sample_size=1., df_red=True), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-90', WithCIAgg(EMQ(newLR()), confidence_level=0.90), {}), # wrap_hyper(logreg_grid)),
    # ('KDEy',  KDEyML(newLR()), {**wrap_hyper(logreg_grid), **{'bandwidth': np.logspace(-4, np.log10(0.2), 20)}}),
]


def add_conf_eval_metric(df):
    # adds an evaluation metric for measuring the goodness of a conf region    

    within = df['within'].values
    proportion = df['proportion'].values

    # in this case, we consider a good confidence region one that has good 
    # coverage ('within') and covers a small proportion of the simplex (1-'proportion')
    # and this is implemented in terms of the harmonic mean
    def armonic_mean(a, b):
        eps = 1e-8
        return 2*a*b/(a+b+eps)
    
    df['harmonicconf'] = armonic_mean(within, 1-proportion)
    
    return df