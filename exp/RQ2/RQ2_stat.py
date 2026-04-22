import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

file_path = "../../results/ablation_result.csv"
sheets = ['balance', 'f_measure', 'auc', 'g_mean']
sheet_names = {'balance': 'Balance', 'f_measure': 'F-measure', 'auc': 'AUC', 'g_mean': 'g-mean'}
tolerance = 1e-3

results_by_metric = {}
for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    data_without_avg = df.iloc[:-1, :]  # remove average row

    n_datasets = len(data_without_avg)
    metric_results = {}

    comparisons = [
        ('CEOS vs None', data_without_avg.iloc[:, 4].values, data_without_avg.iloc[:, 1].values),
        ('CEOS vs CEOS_CS', data_without_avg.iloc[:, 4].values, data_without_avg.iloc[:, 2].values),
        ('CEOS vs CEOS_OS', data_without_avg.iloc[:, 4].values, data_without_avg.iloc[:, 3].values)
    ]

    for comp_name, data1, data2 in comparisons:
        stat, p_value = wilcoxon(data1, data2)

        wins = np.sum(data1 > data2 + tolerance)
        losses = np.sum(data1 < data2 - tolerance)
        draws = n_datasets - wins - losses

        metric_results[comp_name] = {
            'wdl': f"{wins}/{draws}/{losses}",
            'p_value': p_value
        }

    results_by_metric[sheet_names[sheet]] = metric_results

print("Statistical Performance of CEOS and Compared Methods")
print("=" * 100)

headers = ["CEOS vs.", "W/D/L", "p-Value"] * 4
print(f"{'':<20} {'balance':<20} {'F-measure':<23} {'AUC':<23} {'g-mean':<20}")
print(
    f"{'':<15} {'W/D/L':<10} {'p-Value':<10} {'W/D/L':<10} {'p-Value':<10} {'W/D/L':<10} "
    f"{'p-Value':<10} {'W/D/L':<10} {'p-Value':<10}")
print("-" * 100)

comparison_methods = ['None', 'CEOS_CS', 'CEOS_OS']

for i, method in enumerate(comparison_methods):
    comp_name = f"CEOS vs {method}"
    row = [comp_name]

    for metric in ['Balance', 'F-measure', 'AUC', 'g-mean']:
        if comp_name in results_by_metric[metric]:
            result = results_by_metric[metric][comp_name]
            row.extend([result['wdl'], f"{result['p_value']:.4f}"])
        else:
            row.extend(["", ""])

    print(
        f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} "
        f"{row[5]:<10} {row[6]:<10} {row[7]:<10} {row[8]:<10}")

print("=" * 100)
