import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['mathtext.default'] = 'regular'

file_path = '../../results/ablation_result.csv'
sheets = ['balance', 'f_measure', 'auc', 'g_mean']
sheets_draw = ['balance', 'F$_1$', 'AUC', 'G-mean']
data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
colors = ['#d4eae0', '#f3d8e1', '#8abcd1', '#eb507e']
methods = data[sheets[0]].columns[1:]

plt.figure(figsize=(8, 5))

y_min, y_max = 0, 1

for i, sheet in enumerate(sheets):
    for j, method in enumerate(methods):
        method_data = data[sheet][method].iloc[:-1].copy()  # remove the last row which is the average

        box = plt.boxplot(method_data,
                          showfliers=False,
                          positions=[i * (len(methods) + 1) + j + 1],
                          boxprops=dict(color='black', linewidth=1.5),
                          medianprops=dict(color='black', linewidth=2),
                          meanprops=dict(marker='^', markerfacecolor='green',
                                         markeredgecolor='green', markersize=6),
                          showmeans=True,
                          patch_artist=True,
                          widths=0.6)

        for patch in box['boxes']:
            patch.set_facecolor(colors[j])

x_ticks_positions = [(i * (len(methods) + 1) + (len(methods) / 2) + 0.5) for i in range(len(sheets))]
plt.xticks(x_ticks_positions, sheets_draw)

plt.xlim(0.2, x_ticks_positions[-1] + (len(methods) / 2) + 0.5)
plt.ylim(y_min, y_max)
plt.yticks(np.arange(0, 1.1, 0.2), ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

method_names = ['None', 'CEOS$_{CS}$', 'CEOS$_{OS}$', 'CEOS']
legend_handles = [mp.Patch(color=colors[i], label=method_names[i]) for i in range(len(methods))]
plt.legend(handles=legend_handles, bbox_to_anchor=(0.5, 1.11),
           loc='upper center', ncol=len(methods), frameon=False)

plt.subplots_adjust(top=0.926,
                    bottom=0.076,
                    left=0.063,
                    right=0.988,
                    hspace=0.215,
                    wspace=0.2)
plt.show()
