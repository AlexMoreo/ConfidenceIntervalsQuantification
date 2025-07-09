import os
import pandas as pd
from pathlib import Path
from commons import *
from glob import glob


def collect_results(result_dir, quantifiers):
    row_entries = []
    for q_name in quantifiers:
        files = glob(os.path.join(result_dir, f'{q_name}*'))
        for file in files:
            series = {}
            filename = Path(file).name
            filename = filename.replace('-95','')
            method_id, *dataset = filename.split('_')
            dataset='_'.join(dataset)

            conf_region = None
            if 'mInt' in method_id:
                conf_region = 'CI'
            elif 'mReg' in method_id:
                conf_region = 'CE'
            elif 'mClr' in method_id:
                conf_region = 'CT'
            else:
                conf_region = '-'

            bootstrap = None
            if 'tr500-te1' in method_id:
                bootstrap = 'model-based'
            elif 'tr1-te500' in method_id:
                bootstrap = 'population-based'
            elif 'tr100-te100' in method_id:
                bootstrap = 'combined'
            else:
                bootstrap = '-'

            report = qp.util.load_report(file)
            report_means = report.mean(numeric_only=True)

            series['q_name'] = q_name
            series['bootstrap'] = bootstrap
            series['conf-region'] = conf_region
            series['dataset'] = dataset
            series['tr-time'] = report_means['tr_time']
            series['te-time'] = report_means['te_time']

            row_entries.append(series)

    report = pd.DataFrame.from_records(row_entries)
    pivot = report.pivot_table(
        index=['bootstrap', 'conf-region'],
        values=['tr-time', 'te-time'],
        aggfunc='mean'
    )
    desired_order = ['tr-time', 'te-time']
    pivot = pivot[desired_order]

    bootstrap_order = ['-', 'population-based', 'model-based', 'combined']
    conf_region_order = ['-', 'CI', 'CE', 'CT']

    pivot = pivot.reset_index()
    pivot['bootstrap'] = pd.Categorical(pivot['bootstrap'], categories=bootstrap_order, ordered=True)
    pivot['conf-region'] = pd.Categorical(pivot['conf-region'], categories=conf_region_order, ordered=True)

    pivot = pivot.sort_values(['bootstrap', 'conf-region'])

    values = pivot.values
    tr_times = values[:,2]
    te_times = values[:,3]
    tr_orig_time, tr_pop_time, tr_mod_time, tr_comb_time = tr_times
    te_orig_time, te_pop_time, te_mod_time, te_comb_time = te_times
    tr_naive_pop_time = tr_orig_time # 1 training
    tr_pop_time = tr_orig_time # there is only 1 training, we hide possible mismatches between both metrics, which should be the same
    te_naive_pop_time = te_orig_time * 500 # 500 tests
    tr_naive_mod_time = tr_orig_time * 500 # 500 trainings
    te_naive_mod_time = te_orig_time * 500 # although there is no bootstrap in test, there are 500 (trained) models to test
    tr_naive_comb_time = tr_orig_time * 100
    te_naive_comb_time = te_orig_time * (100*100)

    def rel_red(new, old):
        rel = 100*(old-new)/old
        sign = '-' if rel>0 else '+'
        rel = abs(rel)
        return f'({sign}{rel:.2f}\%)'

    tab = ['\\begin{table}']
    tab.append('\\begin{tabular}{llcccc}')
    tab.append('\\toprule')
    tab.append('Bootstrap & Implem. & Train time (s) & (\%Rel.Red) & Test time (s) & (\%Rel.Red)\\\\')
    tab.append('\midrule')
    tab.append('\\toprule')
    tab.append(f'None & \\texttt{{QuaPy}} & {tr_orig_time:.3f} & -- & {te_orig_time:.3f} & -- \\\\ \hline')
    tab.append(f'\multirow{{2}}{{*}}{{population-based}} & Naive & {tr_naive_pop_time:.3f} & -- & {te_naive_pop_time:.3f} & -- \\\\ ')
    tab.append(f'  & Ours & {tr_pop_time:.3f} & {rel_red(tr_pop_time, tr_naive_pop_time)} & {te_pop_time:.3f} & {rel_red(te_pop_time, te_naive_pop_time)} \\\\ \hline')
    tab.append(f'\multirow{{2}}{{*}}{{model-based}} & Naive & {tr_naive_mod_time:.3f} & -- & {te_naive_mod_time:.3f} & -- \\\\ ')
    tab.append(f'  & Ours & {tr_mod_time:.3f} & {rel_red(tr_mod_time, tr_naive_mod_time)} & {te_mod_time:.3f} & {rel_red(te_mod_time, te_naive_mod_time)} \\\\ \hline')
    tab.append(f'\multirow{{2}}{{*}}{{combined}} & Naive & {tr_naive_comb_time:.3f} & -- & {te_naive_comb_time:.3f} & -- \\\\ ')
    tab.append(f'  & Ours & {tr_comb_time:.3f} & {rel_red(tr_comb_time, tr_naive_comb_time)} & {te_comb_time:.3f} & {rel_red(te_comb_time, te_naive_comb_time)} \\\\ ')
    tab.append('\\bottomrule')
    tab.append('\end{tabular}')
    tab.append('\caption{Summary of clocked times grouped by bootstrap.}')
    tab.append('\label{tab:results}')
    tab.append('\end{table}')

    return '\n'.join(tab)


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

    for DATASETS, folder in [
        (MULTICLASS_DATASETS, 'ucimulti'),
        (BINARY_DATASETS, 'binary'),
    ]:
        result_dir = f'results_fake/{folder}/{USE_PROTOCOL}'

        quantifiers = ['ACC', 'PACC', 'DM']

        latex_table = collect_results(result_dir, quantifiers)

        qp.util.save_text_file(f'tables_times/{folder}.tex', latex_table)
