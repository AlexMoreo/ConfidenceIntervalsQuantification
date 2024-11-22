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
            if bootstrap != '-':
                report = add_conf_eval_metric(report)
            report_means = report.mean(numeric_only=True)

            series['q_name'] = q_name
            series['bootstrap'] = bootstrap
            series['conf-region'] = conf_region
            series['dataset'] = dataset
            series['mae'] = report_means['mae']
            series['coverage'] = report_means['within'] if bootstrap != '-' else 0
            series['amplitude'] = report_means['proportion']  if bootstrap != '-' else 0
            series['goodness'] = report_means['goodness']  if bootstrap != '-' else 0

            row_entries.append(series)

    report = pd.DataFrame.from_records(row_entries)
    pivot = report.pivot_table(
        index=['bootstrap', 'conf-region'],
        values=['mae', 'coverage', 'amplitude', 'goodness'],
        aggfunc='mean'
    )
    desired_order = ['mae', 'coverage', 'amplitude', 'goodness']
    pivot = pivot[desired_order]

    bootstrap_order = ['-', 'population-based', 'model-based', 'combined']
    conf_region_order = ['-', 'CI', 'CE', 'CT']

    pivot = pivot.reset_index()
    pivot['bootstrap'] = pd.Categorical(pivot['bootstrap'], categories=bootstrap_order, ordered=True)
    pivot['conf-region'] = pd.Categorical(pivot['conf-region'], categories=conf_region_order, ordered=True)

    pivot = pivot.sort_values(['bootstrap', 'conf-region'])

    latex_table = pivot.to_latex(
        index=False,  # No incluir el índice de pandas en la tabla
        float_format="%.3f",  # Formato para los valores numéricos
        caption="Summary of results grouped by bootstrap and confidence region.",
        label="tab:results"
    )

    return latex_table




if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

    for DATASETS, folder in [
        (MULTICLASS_DATASETS, 'ucimulti'),
        (BINARY_DATASETS, 'binary'),
    ]:
        result_dir = f'results/{folder}/{USE_PROTOCOL}'

        quantifiers = ['ACC', 'PACC', 'DM']

        latex_table = collect_results(result_dir, quantifiers)

        qp.util.save_text_file(f'tables_summary/{folder}.tex', latex_table)
