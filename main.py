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


assert USE_PROTOCOL in ['upp', 'npp'], 'wrong protocol'

newProtocol = UPP if USE_PROTOCOL=='upp' else NPP


def job(args):
    dataset, method_name, quantifier, global_result_path = args
    print('init', dataset)

    local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')
    if os.path.exists(local_result_path):
        # if the file already exists, returns it
        print(f'result file {local_result_path} already exist; skipping')
        report = qp.util.load_report(local_result_path)
    else:
        # otherwise, generates and stores it
        print('Init: ', local_result_path)
        with qp.util.temp_seed(SEED):

            # load dataset
            # ------------
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
            t_init = time()
            row_entries = []
            protocol = newProtocol(test, repeats=N_BAGS_TEST)
            pre_classifications = quantifier.classify(test.instances)
            protocol.on_preclassified_instances(pre_classifications, in_place=True)
            errs, success, proportions = [], [], []
            for i, (sample, true_prev) in enumerate(protocol()):
                if isinstance(quantifier, WithCIAgg):
                    pred_prev, confidence_region = quantifier.aggregate_ci(sample)
                    err_mae = qp.error.ae(true_prev, pred_prev)
                    err_mrae = qp.error.rae(true_prev, pred_prev)
                    is_within = confidence_region.within(true_prev)
                    proportion = confidence_region.simplex_portion()

                    series = {
                        'true-prev': true_prev,
                        'estim-prev': pred_prev,
                        'mae': err_mae,
                        'mrae': err_mrae,
                        'within': is_within,
                        'proportion': proportion
                    }
                    row_entries.append(series)

                    errs.append(err_mae)
                    success.append(is_within * 1)
                    proportions.append(proportion)
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

                print(f'[{(i + 1) / protocol.total()}] '
                      f'MAE={np.mean(errs):.4f} '
                      f'Success={100*np.mean(success):.2f}% '
                      f'Proportion={100*np.mean(proportions):.3f}%')
                
            # prepare report
            # --------------
            report = pd.DataFrame.from_records(row_entries)

            test_time = time() - t_init
            report['tr_time'] = train_time
            report['te_time'] = test_time
            report.to_csv(local_result_path)

    return report


def run_experiments(result_dir):

    os.makedirs(result_dir, exist_ok=True)
    
    global_result_path = f'{result_dir}/allmethods.csv'
    with open(global_result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\tSUCCESS\tPROPORTION\tTR-TIME\tTE-TIME\n')

        for method_name, quantifier in METHODS:        
            reports = qp.util.parallel(
                job, 
                [(dataset, method_name, quantifier, global_result_path) for dataset in DATASETS], 
                n_jobs=N_JOBS, 
                asarray=False
            )

            for report, dataset in zip(reports, DATASETS):
                means = report.mean(numeric_only=True)
                if isinstance(quantifier, WithCIAgg):
                    csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["within"]:.2f}\t{means["proportion"]:.3f}\t{means["tr_time"]:.3f}\t{means["te_time"]:.3f}\n')
                else:
                    csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{-1}\t{-1}\t{means["tr_time"]:.3f}\t{means["te_time"]:.3f}\n')
                csv.flush()

    return global_result_path



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE
    qp.environ['N_JOBS'] = N_JOBS

    for DATASETS, folder in [
            (MULTICLASS_DATASETS, 'ucimulti'),
            # (BINARY_DATASETS, 'binary'),
        ]:
        result_dir = f'results/{folder}/{USE_PROTOCOL}'        
        global_result_path = run_experiments(result_dir)
        show_results(global_result_path)

