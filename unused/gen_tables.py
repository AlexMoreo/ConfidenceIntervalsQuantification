import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import quapy as qp
from pathlib import Path
from commons import *
from submodulos.result_table.src.table import Table


def show_results(result_path):
    import pandas as pd
    df = pd.read_csv(result_path, sep='\t')
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


def gen_pdf_tables(tables, tables_path):
    os.makedirs(Path(tables_path).parent, exist_ok=True)
    tables= [table for table in tables.values()]

    method_replace = {}
    for q_name in ['ACC', 'PACC', 'DM', 'SLD']:
        for bootstrap in ['tr1-te500', 'tr500-te1', 'tr100-te100']:
            bootstrap_rep_dict = {
                'tr1-te500': 'P',
                'tr500-te1': 'M',
                'tr100-te100': 'B'
            }
            bootstrap_rep = bootstrap_rep_dict[bootstrap]
            for region in ['region', 'clr', 'intervals']:
                region = region[:3].title()
                region_rep_dict = {
                    'mReg': 'CE',
                    'mClr': 'CT',
                    'mInt': 'CI'
                }
                region_rep = region_rep_dict['m'+region]
                method_replace[f'{q_name}-95-{bootstrap}-m{region}'] = f'{q_name}-{bootstrap_rep}-{region_rep}'
                method_replace[f'{q_name}-{bootstrap}-m{region}'] = f'{q_name}-{bootstrap_rep}-{region_rep}'

    Table.LatexPDF(tables_path, tables, method_replace=method_replace, verbose=False, clean=False, resizebox=True)


def collect_results(result_dir, tables):

    global_result_path = f'{result_dir}/allmethods.csv'
    with open(global_result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\tSUCCESS\tPROPORTION\tTR-TIME\tTE-TIME\n')

        for method_name, _ in METHODS:        
            for dataset in DATASETS:
                print('init', dataset)

                local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')

                if os.path.exists(local_result_path):
                    print(f'result file {local_result_path} already exist; skipping')
                    report = qp.util.load_report(local_result_path)
                    report = add_conf_eval_metric(report)
                    for metric, table in tables.items():
                        table.add(benchmark=dataset, method=method_name, v=report[metric])

                    means = report.mean(numeric_only=True)
                    try:
                        csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["within"]:.2f}\t{means["proportion"]:.3f}\t{means["tr_time"]:.3f}\t{means["te_time"]:.3f}\n')
                    except KeyError as e:
                        print(f'table for {method_name} in {dataset} has missing value {e}')
                        os.remove(local_result_path)
                    csv.flush()
    
    return global_result_path



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

    for DATASETS, folder in [
            (BINARY_DATASETS, 'binary'), 
            (MULTICLASS_DATASETS, 'ucimulti')
        ]:
        result_dir = f'results/{folder}/{USE_PROTOCOL}' 
    
        tables = {
            'mae': Table('mae'),
            # 'mrae': Table('mrae'),
            'within': Table('success'),
            'proportion': Table('proportion'),
            'harmonicconf': Table('harmonicconf')
        }

        tables['mae'].format.show_std = False
        tables['mae'].format.remove_zero = True

        tables['within'].format.show_std = False
        tables['within'].format.mean_prec = 2
        tables['within'].format.stat_test = None
        tables['within'].format.lower_is_better = False
        tables['within'].format.remove_zero = True

        tables['proportion'].format.show_std = False
        tables['proportion'].format.mean_prec = 2
        tables['proportion'].format.stat_test = None
        tables['proportion'].format.lower_is_better = True
        tables['proportion'].format.remove_zero = True
        # tables['harmonicconf'].format.lower_is_better = False
        # tables['critical'].format.show_std = False

        global_result_path = collect_results(result_dir, tables)

        show_results(global_result_path)
        gen_pdf_tables(tables, tables_path=f'./tables/{folder}/{USE_PROTOCOL}/dm.pdf')
