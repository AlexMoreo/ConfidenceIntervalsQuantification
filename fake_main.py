import numpy as np
import pandas as pd
import quapy as qp
import quapy.functional as F
from quapy.protocol import UPP, NPP
from commons import *
from gen_tables import show_results
from model import WithCIAgg
import pickle
import os
from time import time
from pathlib import Path
from tqdm import tqdm


assert USE_PROTOCOL in ['upp', 'npp'], 'wrong protocol'

newProtocol = UPP if USE_PROTOCOL=='upp' else NPP


def job(args):
    dataset, method_name, quantifier, result_dir = args
    print('init', dataset)

    local_result_path = os.path.join(result_dir, method_name + '_' + dataset + '.dataframe')
    if os.path.exists(local_result_path):
        # if the file already exists, returns it
        print(f'result file {local_result_path} already exist; skipping')
        report = qp.util.load_report(local_result_path)
    else:
        # otherwise, generates and stores it
        print('Init: ', local_result_path)
        with qp.util.temp_seed(SEED):

            # load dataset            # ------------
            if dataset in BINARY_DATASETS:
                loader = qp.datasets.fetch_UCIBinaryDataset
            elif dataset in MULTICLASS_DATASETS:
                loader = qp.datasets.fetch_UCIMulticlassDataset

            try:                
                data = loader(dataset, verbose=True)
            except Exception as e:
                print(f'There was an error loading {dataset}: {e}')                

            if np.isnan(data.training.X).any():
                print(f'trainin set {dataset} contain nan values!')
                raise ValueError()
            if np.isnan(data.test.X).any():
                print(f'test set {dataset} contain nan values!')
                raise ValueError()

            
            train, test = data.train_test

            # training
            # ------------
            t_init = time()
            quantifier.fit(train)
            train_time = time() - t_init

            # test
            # ------------
            te_time = 0
            row_entries = []
            protocol = newProtocol(test, repeats=N_BAGS_TEST_FAKE)
            t_init = time()
            pre_classifications = quantifier.classify(test.instances)
            te_time_increment = time()-t_init
            te_time += te_time_increment

            protocol.on_preclassified_instances(pre_classifications, in_place=True)
            errs = []
            pbar = tqdm(protocol(), total=N_BAGS_TEST_FAKE)
            for i, (sample, true_prev) in enumerate(pbar):
                if isinstance(quantifier, WithCIAgg):
                    t_init=time()
                    pred_prev, confidence_region = quantifier.aggregate_ci(sample)
                    err_mae = qp.error.ae(true_prev, pred_prev)
                    err_mrae = qp.error.rae(true_prev, pred_prev)
                    te_time_increment = time() - t_init
                    te_time += te_time_increment

                    series = {
                        'true-prev': true_prev,
                        'estim-prev': pred_prev,
                        'mae': err_mae,
                        'mrae': err_mrae,
                    }
                    row_entries.append(series)

                    errs.append(err_mae)
                else:
                    pred_prev = quantifier.aggregate(sample)
                    err_mae = qp.error.ae(true_prev, pred_prev)
                    err_mrae = qp.error.rae(true_prev, pred_prev)

                    series = {
                        'true-prev': true_prev,
                        'estim-prev': pred_prev,
                        'mae': err_mae,
                        'mrae': err_mrae,
                    }
                    row_entries.append(series)

                    errs.append(err_mae)

                pbar.set_description(f'{train_time=:.4f}\t{te_time=:.4f}')

            # prepare report
            # --------------
            report = pd.DataFrame.from_records(row_entries)

            test_time = te_time / N_BAGS_TEST_FAKE
            report['tr_time'] = train_time
            report['te_time'] = test_time
            report.to_csv(local_result_path)

    return report


def run_experiments(result_dir):

    os.makedirs(result_dir, exist_ok=True)
    
    for method_name, quantifier in METHODS:
        reports = qp.util.parallel(
            job,
            [(dataset, method_name, quantifier, result_dir) for dataset in DATASETS],
            n_jobs=N_JOBS,
            asarray=False
        )




if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE
    qp.environ['N_JOBS'] = N_JOBS

    N_BAGS_TEST_FAKE = 10

    for DATASETS, folder in [
            (MULTICLASS_DATASETS, 'ucimulti'),
            (BINARY_DATASETS, 'binary'),
        ]:
        result_dir = f'results_fake/{folder}/{USE_PROTOCOL}'
        run_experiments(result_dir)

