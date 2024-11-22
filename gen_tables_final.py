import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import quapy as qp
from pathlib import Path
from commons import *
from submodulos.result_table.src.table import Table


def gen_pdf_tables(tables, tables_path, method_prefix):
    os.makedirs(Path(tables_path).parent, exist_ok=True)

    tab = "\\begin{tabular}{|c|c|" +"ccc|"*9+ "}"
    tab += '\n'
    tab += '\cline{2-29} \n'
    tab += '\multicolumn{1}{c}{ } & \multicolumn{28}{|c|}{'+method_prefix+'} \\\\'
    tab += '\cline{2-29} \n'
    tab += '\multicolumn{1}{c}{ } & \multicolumn{1}{|c}{ - } & \multicolumn{9}{|c}{Population-based} & \multicolumn{9}{|c}{Model-based} & \multicolumn{9}{|c|}{Combined} \\\\'
    tab += '\cline{2-29} \n'
    tab += '\multicolumn{1}{c|}{ } & \multicolumn{1}{|c|}{ - } ' + (' & \multicolumn{3}{c|}{ CI } & \multicolumn{3}{c|}{ CE } & \multicolumn{3}{c|}{ CT } ')*3 + '\\\\'
    tab += '\cline{2-29} \n'
    tab += '\multicolumn{1}{c|}{ } & \multicolumn{1}{|c|}{ MAE }' + (' & MAE & $\mathcal{C}$ & $\mathcal{A}$ ' * 9) + '\\\\'
    tab += '\\hline '


    print('generating final table')
    for benchmark in tables['mae'].benchmarks:

        tab += benchmark.replace('_', '\_')

        tab += ' & ' + tables['mae'].get(benchmark, method_prefix).print()

        for bootstrap in ['95-tr1-te500', '95-tr500-te1', 'tr100-te100']:
            for region in ['intervals', 'region', 'clr']:
                region_prefix = 'm'+region[:3].title()
                orig_col = f'{method_prefix}-{bootstrap}-{region_prefix}'

                error = tables['mae'].get(benchmark, orig_col).print()
                coverage = tables['within'].get(benchmark, orig_col).print()
                amplitude = tables['proportion'].get(benchmark, orig_col).print()

                tab += ' & ' + error
                tab += ' & ' + coverage
                tab += ' & ' + amplitude
        tab += '\\\\ \n'

    tab += '\\hline '

    # add mean values and rank values
    method_order = []
    for bootstrap in ['95-tr1-te500','95-tr500-te1','tr100-te100']:
        for region in ['intervals', 'region', 'clr']:
            region_prefix = 'm' + region[:3].title()
            orig_col = f'{method_prefix}-{bootstrap}-{region_prefix}'
            method_order.append(orig_col)

    tab += 'Mean'


    mae_means = tables['mae'].get_method_means([method_prefix] + method_order)
    tab += ' & ' + mae_means[0].print()

    coverage_means = tables['within'].get_method_means(method_order)
    amplitude_means = tables['proportion'].get_method_means(method_order)
    for maem, covm, ampm in zip(mae_means[1:], coverage_means, amplitude_means):
            tab += ' & ' + maem.print()
            tab += ' & ' + covm.print()
            tab += ' & ' + ampm.print()
    tab += '\\\\ \n'
    tab += '\\hline '

    # tab += 'Rank'
    #
    # mae_means = tables['mae'].get_method_rank_means([method_prefix]+method_order)
    # tab += ' & ' + mae_means[0].print()
    # coverage_means = tables['within'].get_method_rank_means(method_order)
    # amplitude_means = tables['proportion'].get_method_rank_means(method_order)
    # for maem, covm, ampm in zip(mae_means[1:], coverage_means, amplitude_means):
    #         tab += ' & ' + maem.print()
    #         tab += ' & ' + covm.print()
    #         tab += ' & ' + ampm.print()
    # tab += '\\\\ \n'
    # tab += '\\hline '

    tab += '\end{tabular}\n'

    print(tables_path)
    with open(tables_path, 'wt') as foo:
        foo.write(tab)


def gen_goodness_tables(table, tables_path, method_prefix):
    os.makedirs(Path(tables_path).parent, exist_ok=True)
    method_replace={}
    method_order = []
    for bootstrap, boot_rep in [('95-tr1-te500', 'Pop'), ('95-tr500-te1','Mod'), ('tr100-te100', 'Comb')]:
        for region, reg_rep in [('mInt', 'CI'), ('mReg', 'CE'), ('mClr', 'CT')]:
            orig_name = f'{method_prefix}-{bootstrap}-{region}'
            method_order.append(orig_name)
            method_replace[orig_name] = f'{boot_rep}-{reg_rep}'
    # table.latexPDF(tables_path, method_replace=method_replace, method_order=method_order)
    table_lines = table.tabular(path=None, method_replace=method_replace, method_order=method_order, return_lines=True)
    table_lines[0] = table_lines[0].replace('ccc', 'ccc|')
    table_lines.insert(2, '\multicolumn{1}{c|}{} & \multicolumn{3}{c|}{Population-based} & \multicolumn{3}{c|}{Model-based} & \multicolumn{3}{c|}{Combined} \\\\')
    table_lines.insert(3, table_lines[1]) # a cline
    table_lines[4] = table_lines[4].replace('Pop-','')
    table_lines[4] = table_lines[4].replace('Mod-', '')
    table_lines[4] = table_lines[4].replace('Comb-', '')
    table = '\n'.join(table_lines)
    with open(tables_path, 'wt') as foo:
        foo.write(table)
    # print(table_lines)


def collect_results(tables, result_dir, method_prefix):
    for method_name, _ in METHODS:
        if method_name.startswith(method_prefix):
            for dataset in DATASETS:
                local_result_path = os.path.join(result_dir, method_name + '_' + dataset + '.dataframe')

                if os.path.exists(local_result_path):
                    report = qp.util.load_report(local_result_path)
                    if method_name != method_prefix:
                        report = add_conf_eval_metric(report)
                    for metric, table in tables.items():
                        if method_name == method_prefix:
                            if metric not in ['within', 'proportion', 'goodness']:
                                table.add(benchmark=dataset, method=method_name, v=report[metric])
                        else:
                            table.add(benchmark=dataset, method=method_name, v=report[metric])
                else:
                    raise ValueError('Missing' + local_result_path)



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

    for DATASETS, folder in [
        (MULTICLASS_DATASETS, 'ucimulti'),
        (BINARY_DATASETS, 'binary'),
    ]:
        result_dir = f'results/{folder}/{USE_PROTOCOL}'

        for method_prefix in ['ACC', 'PACC', 'DM']:

            tables = {
                'mae': Table('mae'),
                'within': Table('success'),
                'proportion': Table('proportion'),
                'goodness': Table('goodness_'+method_prefix)
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

            tables['goodness'].format.lower_is_better = False
            tables['goodness'].format.show_std = False
            tables['goodness'].format.with_rank_mean = False

            collect_results(tables, result_dir, method_prefix)

            pdf_path = f'./tables_final/{folder}/{USE_PROTOCOL}/{method_prefix}_tab.tex'
            gen_pdf_tables(tables, tables_path=pdf_path, method_prefix=method_prefix)

            pdf_path2 = f'./tables_goodness/{folder}/{USE_PROTOCOL}/{method_prefix}_goodness_tab.tex'
            gen_goodness_tables(tables['goodness'], tables_path=pdf_path2, method_prefix=method_prefix)
