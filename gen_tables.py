import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import quapy as qp
from pathlib import Path


from main import METHODS, USE_PROTOCOL
from submodulos.result_table.src.table import Table



def show_results(result_path, tables, tables_path=f'./tables/{USE_PROTOCOL}/main.pdf'):
    import pandas as pd
    df = pd.read_csv(result_path + '.csv', sep='\t')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)  # Ajustar el ancho máximo
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE"], margins=True)
    print(pv)
    # pv = df.pivot_table(index='Dataset', columns="Method", values=["MRAE"], margins=True)
    # print(pv)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["SUCCESS"], margins=True)
    print(pv)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["PROPORTION"], margins=True)
    print(pv)
    # pv = df.pivot_table(index='Dataset', columns="Method", values=["TE-TIME"], margins=True)
    # print(pv)

    os.makedirs(Path(tables_path).parent, exist_ok=True)
    tables= [table for table in tables.values()]

    method_replace = {
        'PACC-95': 'PACC-$\\frac{|Te|}{2}$',
        'PACC-95-1': 'PACC-$|Te|$',
        'PACC-95-1-ddof-1': 'PACC-$|Te|$(-1)',
        'SLD-95': 'SLD-$\\frac{|Te|}{2}$',
        'SLD-95-1': 'SLD-$|Te|$',
        'SLD-95-1-ddof-1': 'SLD-$|Te|$(-1)',
    }

    Table.LatexPDF(tables_path, tables, method_replace=method_replace, verbose=True, clean=True, resizebox=True)


def collect_results(method_name, tables):

    print('Init method', method_name)

    with open(global_result_path + '.csv', 'at') as csv:
        for dataset in qp.datasets.UCI_MULTICLASS_DATASETS:
            print('init', dataset)

            # run_experiment(global_result_path, method_name, quantifier, param_grid, dataset)
            local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')

            if os.path.exists(local_result_path):
                print(f'result file {local_result_path} already exist; skipping')
                report = qp.util.load_report(local_result_path)
                for metric, table in tables.items():
                    tables[metric].add(benchmark=dataset, method=method_name, v=report[metric])

                means = report.mean(numeric_only=True)
                csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["within"]:.2f}\t{means["proportion"]:.3f}\t{means["tr_time"]:.3f}\t{means["te_time"]:.3f}\n')
                csv.flush()



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 500
    qp.environ['N_JOBS'] = -1
    n_bags_val = 100
    n_bags_test = 500
    result_dir = f'results/ucimulti/{USE_PROTOCOL}'

    os.makedirs(result_dir, exist_ok=True)

    tables = {
        'mae': Table('mae'),
        # 'mrae': Table('mrae'),
        'within': Table('success'),
        'critical': Table('critical'),
    }

    # tables['mae'].format.stat_test = None
    tables['within'].format.show_std = False
    tables['within'].format.mean_prec = 2
    tables['within'].format.stat_test = None
    tables['within'].format.lower_is_better = False
    tables['critical'].format.show_std = False


    global_result_path = f'{result_dir}/allmethods'
    with open(global_result_path + '.csv', 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\tSUCCESS\tPROPORTION\tTR-TIME\tTE-TIME\n')

    for method_name, _, _ in METHODS:
        collect_results(method_name, tables)

    show_results(global_result_path, tables)
