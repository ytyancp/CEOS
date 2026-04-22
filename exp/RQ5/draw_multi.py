import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors


file_path = "../../results/multi_results_10.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)
matplotlib.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.sans-serif'] = ['Times New Roman']

classifiers = ['RF', 'MLP', 'KNN', 'LR']
metric_names = {'balance': 'balance', 'f_measure': 'F$_1$', 'auc': 'AUC', 'g_mean': 'G-mean'}
metrics = list(metric_names.keys())

first_sheet = list(all_sheets.values())[0]
methods = first_sheet.columns[1:].tolist()
methods = [method.replace('TCA+CEOS', 'CEOS$_{TCA}$').replace('TCA++CEOS', 'CEOS$_{TCA+}$').replace('NNFilter', 'NN-filter')
           for method in methods]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()

for metric_idx, metric in enumerate(metrics):
    ax = axes[metric_idx]
    metric_data = []

    for classifier in classifiers:
        sheet_name = f"{classifier}_{metric}"
        df = all_sheets[sheet_name]
        data_values = df.iloc[:-1, 1:].values
        ranks = pd.DataFrame(data_values).rank(axis=1, ascending=False)
        mean_ranks = ranks.mean(axis=0).values
        metric_data.append(mean_ranks)

    metric_matrix = np.array(metric_data).T
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue_red', colors)

    im = ax.imshow(metric_matrix, cmap=cmap, aspect='auto', interpolation='nearest')

    ax.set_xticks(np.arange(len(classifiers)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(classifiers, fontsize=15, fontweight='bold')
    if metric_idx % 2 == 0:
        ax.set_yticklabels(methods, fontsize=15, fontweight='bold', ha='left')
        ax.tick_params(axis='y', pad=75)
    else:
        ax.set_yticklabels(())

    ax.set_title(f'{metric_names[metric]}', fontsize=15, fontweight='bold')

    for i in range(len(methods)):
        for j in range(len(classifiers)):
            text = ax.text(j, i, f'{metric_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=18)

    ax.set_xticks(np.arange(-0.5, len(classifiers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='both', length=0)

fig.subplots_adjust(right=0.8)
plt.tight_layout()
plt.show()
