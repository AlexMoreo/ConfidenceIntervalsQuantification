import numpy as np
import pandas as pd
import quapy as qp
import quapy.functional as F
from quapy.protocol import UPP, NPP
from quapy.method.aggregative import PACC, EMQ, KDEyML
from model import WithCIAgg, simplex_proportion_covered
import pickle
import os
from time import time
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from quapy.model_selection import GridSearchQ
from pathlib import Path

SEED = 1

USE_PROTOCOL = 'upp'

assert USE_PROTOCOL in ['upp', 'npp'], 'wrong protocol'

newProtocol= UPP if USE_PROTOCOL=='upp' else NPP


def newLR():
    return LogisticRegression(max_iter=3000)


# typical hyperparameters explored for Logistic Regression
logreg_grid = {
    'C': np.logspace(-3,3,7),
    'class_weight': [None, 'balanced']
}


def wrap_hyper(classifier_hyper_grid: dict):
    return {'classifier__' + k: v for k, v in classifier_hyper_grid.items()}


METHODS = [
    # ('PACC-99', WithCIAgg(PACC(newLR()), confidence_level=0.99), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-0.5-I', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=0.5), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-1-I', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1.), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-0.5-CLR', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=0.5, transform='clr'), {}), # wrap_hyper(logreg_grid)),
    ('PACC-95-1-CLR', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., transform='clr'), {}), # wrap_hyper(logreg_grid)),
    # ('PACC-95-1-ddof-1-optim', WithCIAgg(PACC(newLR()), confidence_level=0.95, sample_size=1., df_red=True), wrap_hyper(logreg_grid)), # wrap_hyper(logreg_grid)),
    # ('PACC-90', WithCIAgg(PACC(newLR()), confidence_level=0.90), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-99', WithCIAgg(EMQ(newLR()), confidence_level=0.99), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-95', WithCIAgg(EMQ(newLR()), confidence_level=0.95, sample_size=0.5), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-95-1', WithCIAgg(EMQ(newLR()), confidence_level=0.95, sample_size=1.), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-95-1-ddof-1', WithCIAgg(EMQ(newLR()), confidence_level=0.95, sample_size=1., df_red=True), {}), # wrap_hyper(logreg_grid)),
    # ('SLD-90', WithCIAgg(EMQ(newLR()), confidence_level=0.90), {}), # wrap_hyper(logreg_grid)),
    # ('KDEy',  KDEyML(newLR()), {**wrap_hyper(logreg_grid), **{'bandwidth': np.logspace(-4, np.log10(0.2), 20)}}),
]



def show_results(result_path):
    import pandas as pd
    df = pd.read_csv(result_path + '.csv', sep='\t')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)  # Ajustar el ancho máximo
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE"], margins=True)
    print(pv)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MRAE"], margins=True)
    print(pv)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["SUCCESS"], margins=True)
    print(pv)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["TR-TIME"], margins=True)
    print(pv)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["TE-TIME"], margins=True)
    print(pv)


def job(args):
    dataset, quantifier = args
    print('init', dataset)

    local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')
    if os.path.exists(local_result_path):
        print(f'result file {local_result_path} already exist; skipping')
        report = qp.util.load_report(local_result_path)
    else:
        with qp.util.temp_seed(SEED):

            data = qp.datasets.fetch_UCIMulticlassDataset(dataset, verbose=True)
            train, test = data.train_test

            if len(param_grid) == 0:
                t_init = time()
                quantifier.fit(train)
                train_time = time() - t_init
            else:
                # model selection (train)
                train, val = train.split_stratified(random_state=SEED)
                protocol = newProtocol(val, repeats=n_bags_val)
                modsel = GridSearchQ(
                    quantifier, param_grid, protocol, refit=True, n_jobs=-1, verbose=1, error='mae'
                )
                t_init = time()
                try:
                    modsel.fit(train)
                    print(f'best params {modsel.best_params_}')
                    print(f'best score {modsel.best_score_}')
                    quantifier = modsel.best_model()
                except:
                    print('something went wrong... trying to fit the default model')
                    quantifier.fit(train)
                train_time = time() - t_init

            # test
            t_init = time()

            row_entries = []
            protocol = newProtocol(test, repeats=n_bags_test)
            pre_classifications = quantifier.classify(test.instances)
            protocol.on_preclassified_instances(pre_classifications, in_place=True)
            errs, success, proportions = [], [], []
            for i, (sample, true_prev) in enumerate(protocol()):
                pred_prev, confidence_region = quantifier.aggregate_ci(sample)
                err_mae = qp.error.ae(true_prev, pred_prev)
                err_mrae = qp.error.rae(true_prev, pred_prev)
                is_within = confidence_region.within(true_prev)
                proportion = simplex_proportion_covered(confidence_region)

                series = {
                    'true-prev': true_prev,
                    'estim-prev': pred_prev,
                    'mae': err_mae,
                    'mrae': err_mrae,
                    'within': is_within,
                    'critical': confidence_region.chi2_critical,
                    'proportion': proportion
                }
                row_entries.append(series)

                errs.append(err_mae)
                success.append(is_within * 1)
                proportions.append(proportion)

                print(f'[{(i + 1) / n_bags_test}] '
                      f'MAE={np.mean(errs):.4f} '
                      f'Success={100*np.mean(success):.2f}% '
                      f'Proportion={100*np.mean(proportions):.3f}%')

            report = pd.DataFrame.from_records(row_entries)

            test_time = time() - t_init
            report['tr_time'] = train_time
            report['te_time'] = test_time
            report.to_csv(local_result_path)
    return report


def run_experiment(method_name, quantifier, param_grid):

    print('Init method', method_name)
    datasets = qp.datasets.UCI_MULTICLASS_DATASETS

    with open(global_result_path + '.csv', 'at') as csv:
        reports = qp.util.parallel(job, [(dataset, quantifier) for dataset in datasets], n_jobs=-1, asarray=False)

        for report, dataset in zip(reports, datasets):
            means = report.mean(numeric_only=True)
            csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["within"]:.2f}\t{means["proportion"]:.3f}\t{means["tr_time"]:.3f}\t{means["te_time"]:.3f}\n')
            csv.flush()



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 500
    qp.environ['N_JOBS'] = -1
    n_bags_val = 100
    n_bags_test = 100
    result_dir = f'results/ucimulti/{USE_PROTOCOL}'

    os.makedirs(result_dir, exist_ok=True)

    global_result_path = f'{result_dir}/allmethods'
    with open(global_result_path + '.csv', 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\tSUCCESS\tPROPORTION\tTR-TIME\tTE-TIME\n')

    for method_name, quantifier, param_grid in METHODS:
        run_experiment(method_name, quantifier, param_grid)

    show_results(global_result_path)

