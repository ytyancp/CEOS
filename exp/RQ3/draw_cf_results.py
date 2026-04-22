import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9

excel_path = "../../results/RQ3/cf_results.xlsx"
validity_df = pd.read_excel(excel_path, sheet_name='validity', index_col=0)
proximity_l1_df = pd.read_excel(excel_path, sheet_name='proximity_l1', index_col=0)
proximity_l2_df = pd.read_excel(excel_path, sheet_name='proximity_l2', index_col=0)
plausibility_df = pd.read_excel(excel_path, sheet_name='plausibility', index_col=0)

validity_df = validity_df.iloc[:-1]  # remove average row
proximity_l1_df = proximity_l1_df.iloc[:-1]
proximity_l2_df = proximity_l2_df.iloc[:-1]
plausibility_df = plausibility_df.iloc[:-1]

methods = list(validity_df.columns)


def calculate_stats(df):
    mean = df.mean()
    std = df.std()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    return mean, std, Q1, Q3


validity_stats = calculate_stats(validity_df)
proximity_l1_stats = calculate_stats(proximity_l1_df)
proximity_l2_stats = calculate_stats(proximity_l2_df)
plausibility_stats = calculate_stats(plausibility_df)

validity_ranks = validity_df.rank(axis=1, ascending=False)
proximity_l1_ranks = proximity_l1_df.rank(axis=1, ascending=True)
proximity_l2_ranks = proximity_l2_df.rank(axis=1, ascending=True)
plausibility_ranks = plausibility_df.rank(axis=1, ascending=True)
mean_rank_df = (validity_ranks + proximity_l1_ranks + proximity_l2_ranks + plausibility_ranks) / 4
mean_rank_stats = calculate_stats(mean_rank_df)

metrics_names = ['Validity ↑', 'Proximity$_{L1}$ ↓', 'Proximity$_{L2}$ ↓', 'Plausibility ↓', 'Mean Rank ↓']
all_stats = [validity_stats, proximity_l1_stats, proximity_l2_stats, plausibility_stats, mean_rank_stats]

colors = ['#f0d6a5', '#7c85ff', '#d081ef', '#6de96d', '#a67b3d', '#ae81a7', '#fe807f'][:len(methods)]
ceos_index = methods.index('CEOS') if 'CEOS' in methods else -1

fig = plt.figure(figsize=(13, 3.2))

# We create two GridSpec: the first 4 subplots use one, and the last one uses one.
gs_left = GridSpec(1, 4, figure=fig, left=0.03, right=0.8, wspace=0.195)
gs_right = GridSpec(1, 1, figure=fig)
gs_right.update(left=0.81, right=0.97)

axes = []
for i in range(4):
    axes.append(fig.add_subplot(gs_left[0, i]))

axes.append(fig.add_subplot(gs_right[0, 0]))

for idx, (ax, metric_name, stats) in enumerate(zip(axes, metrics_names, all_stats)):
    mean, std, Q1, Q3 = stats
    bar_width = 0.8

    for i, method in enumerate(methods):
        ax.bar(i, mean[method], bar_width, color=colors[i], edgecolor=colors[i],
               linewidth=0.8, zorder=3, alpha=0.7)

    for i, method in enumerate(methods):
        mean_val = mean[method]
        std_val = std[method]
        q1_val = Q1[method]
        q1_val = max(0, q1_val)
        q3_val = Q3[method]
        x_center = i

        y_high = mean_val + std_val
        y_low = max(0, mean_val - std_val)

        ax.plot([x_center, x_center], [y_low, y_high],
                color=colors[i], linewidth=2.5, zorder=4)

        ax.plot([x_center - 0.12, x_center + 0.12], [y_high, y_high],
                color=colors[i], linewidth=2.5, zorder=4)

        ax.plot([x_center - 0.12, x_center + 0.12], [y_low, y_low],
                color=colors[i], linewidth=2.5, zorder=4)  # draw lower error bar

        ax.plot(x_center, q3_val, 'o', color=colors[i], markersize=5,
                markeredgecolor=colors[i], markeredgewidth=0.8, zorder=5)
        ax.plot(x_center, q1_val, 'o', color=colors[i], markersize=5,
                markeredgecolor=colors[i], markeredgewidth=0.8, zorder=5)

    ax.set_xticks([])
    ax.set_xticklabels([])

    if idx < 4:
        min_mean = min(mean)
        max_mean = max(mean)
        max_error = max([mean[method] + std[method] for method in methods])
        y_min = 0
        y_max = max_mean * 1.15

        if idx == 0:
            y_min = 0.5
            y_max = 1
        elif idx == 1:
            y_min = 0.5
            y_max = 7
        elif idx == 2:
            y_min = 0.25
            y_max = 2
        else:
            y_max = 2.5

        ax.set_ylim([y_min, y_max])
        mean_range = y_max - y_min

        if mean_range < 1:
            y_ticks = np.linspace(y_min, y_max, 6)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=10, fontweight='bold')
        elif mean_range < 4:
            y_ticks = np.linspace(y_min, y_max, 5)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks], fontsize=10, fontweight='bold')
        else:
            y_ticks = np.linspace(y_min, y_max, 4)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=10, fontweight='bold')

    else:  # mean rank
        all_q1 = [max(0, Q1[method]) for method in methods]
        all_q3 = [Q3[method] for method in methods]
        all_upper = [mean[method] + std[method] for method in methods]

        min_val = min(all_q1)
        max_val = max(all_q3 + all_upper)
        y_min = 1
        y_max = max_val + 1

        ax2 = ax.twinx()
        ax2.set_ylim([y_min, y_max])

        if (max_val - min_val) < 5:
            y_ticks = np.linspace(y_min, y_max, 5)
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=10)
        else:
            y_ticks = np.linspace(y_min, y_max, 4)
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{tick:.0f}' for tick in y_ticks], fontsize=10)

        ax.tick_params(axis='y', labelleft=False, left=False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.2, zorder=1, linewidth=0.2)

    ax.text(0.5, 1.02, metric_name, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    if ceos_index >= 0:
        ceos_mean = mean.iloc[ceos_index]
        ceos_color = colors[ceos_index]
        ax.plot([-0.5, ceos_index], [ceos_mean, ceos_mean],
                color='black', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)

        ax.plot([ceos_index, ceos_index],
                [ceos_mean, ceos_mean + 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])],
                color='black', linewidth=1.0, linestyle='-', alpha=0.8, zorder=2)

        if idx == 0:
            ax.text(-1.95, ceos_mean, f'{ceos_mean:.2f}',
                    fontsize=10, va='center', ha='left', color=ceos_color, fontweight='bold')
        elif idx == 1:
            ax.text(-1.5, ceos_mean, f'{ceos_mean:.1f}',
                    fontsize=10, va='center', ha='center', color=ceos_color, fontweight='bold')
        elif idx == 2:
            ax.text(-1.62, ceos_mean, f'{ceos_mean:.2f}',
                    fontsize=10, va='center', ha='center', color=ceos_color, fontweight='bold')
        elif idx == 3:
            ax.text(-1.61, ceos_mean, f'{ceos_mean:.2f}',
                    fontsize=10, va='center', ha='center', color=ceos_color, fontweight='bold')
        else:
            ax.text(7.9, ceos_mean-0.05, f'{ceos_mean:.2f}',
                    fontsize=10, va='center', ha='right', color=ceos_color, fontweight='bold')


legend_elements = [Rectangle((0, 0), 1, 1, facecolor=colors[i],
                             edgecolor='black', linewidth=0.8) for i in range(len(methods))]

legend = fig.legend(legend_elements, methods,
                    loc='lower center',
                    ncol=len(methods),
                    fontsize=11,
                    frameon=False,
                    bbox_to_anchor=(0.5, -0.03),
                    handlelength=5.5,
                    handleheight=1.3)

plt.tight_layout()
plt.subplots_adjust()

plt.show()
