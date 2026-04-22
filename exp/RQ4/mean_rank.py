import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
classifiers = ['KNN', 'MLP', 'LR', 'RF']
path = '../../results'
sheets = ['balance', 'f_measure', 'auc', 'g_mean']


def calculate_mean_rank(data):
    data_without_avg = data.iloc[:-1, 2:-1].astype(float)
    mean_rank = data_without_avg.apply(lambda row: row.rank(ascending=False), axis=1).mean()

    return mean_rank


all_mean_ranks = {sheet: pd.DataFrame() for sheet in sheets}

for classifier in classifiers:
    file_path = f'{path}\\final_{classifier}.csv'
    dataframes = {}

    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        mean_ranks = calculate_mean_rank(df)
        mean_ranks.name = classifier
        if sheet not in all_mean_ranks:
            all_mean_ranks[sheet] = pd.DataFrame()
        all_mean_ranks[sheet][classifier] = mean_ranks

final_mean_ranks = {sheet: all_mean_ranks[sheet].mean(axis=1) for sheet in sheets}
final_mean_ranks_df = pd.DataFrame(final_mean_ranks).T

fig, ax = plt.subplots(figsize=(9, 5))

num_metrics = len(final_mean_ranks_df.index)
num_methods = len(final_mean_ranks_df.columns)

bar_width = 0.05
x = np.arange(num_metrics)
colors = ['#0072bd', '#d95319', '#edb120', '#7e2f8e', '#77ac30',
          '#4DBEee', '#a2142f', '#bfbf00', '#bf00bf', '#007f00',
          '#4e6594', '#B55489', '#dab3ff', '#c1ddc6', '#00bfBF']
for i, method in enumerate(final_mean_ranks_df.columns):
    a = x + (i * (bar_width + 0.01))
    plt.bar(x + (i * (bar_width + 0.01)), final_mean_ranks_df[method], width=bar_width, label=method, color=colors[i])

plt.xticks(x + bar_width * (num_methods - 1) / 2, final_mean_ranks_df.index)
ax.set_xticklabels(['balance', 'F$_1$', 'AUC', 'G-mean'])
plt.xticks(rotation=0)
x_min, x_max = ax.get_xlim()
plt.xlim(left=x_min + 0.15, right=x_max - 0.15)
plt.ylim(0, final_mean_ranks_df.max().max() + 3.4)
plt.legend(bbox_to_anchor=(0.5, 1.005), loc='upper center', ncol=5, fancybox=True)
plt.tight_layout(pad=0.4)
plt.show()
