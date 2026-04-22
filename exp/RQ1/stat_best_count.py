import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

balance_data = np.array([
    [4, 3, 4, 4],  # iteration 3
    [3, 4, 3, 4],  # iteration 4
    [11, 8, 9, 7],  # iteration 5
    [5, 5, 7, 4],  # iteration 6
    [3, 3, 3, 2],  # iteration 7
    [2, 1, 1, 3],  # iteration 8
    [2, 2, 2, 5],  # iteration 9
    [1, 0, 1, 2],  # iteration 10
    [0, 1, 1, 1],  # iteration 11
    [1, 5, 1, 0]  # iteration 12
])

f1_data = np.array([
    [1, 2, 2, 1],
    [2, 4, 3, 2],
    [11, 7, 10, 10],
    [5, 3, 4, 5],
    [4, 3, 4, 2],
    [3, 2, 3, 3],
    [2, 1, 2, 4],
    [2, 2, 2, 3],
    [1, 1, 2, 1],
    [1, 7, 0, 1]
])

auc_data = np.array([
    [2, 0, 1, 2],
    [2, 2, 2, 1],
    [12, 8, 10, 10],
    [4, 4, 10, 2],
    [5, 3, 4, 3],
    [2, 3, 2, 2],
    [2, 1, 2, 6],
    [1, 2, 0, 2],
    [2, 1, 1, 2],
    [0, 8, 0, 2]
])

g_mean_data = np.array([
    [1, 2, 1, 3],
    [0, 2, 2, 2],
    [11, 8, 11, 9],
    [3, 4, 3, 4],
    [4, 3, 5, 3],
    [3, 2, 2, 2],
    [4, 2, 4, 7],
    [3, 1, 1, 1],
    [1, 3, 2, 0],
    [2, 5, 1, 1]
])

classifiers = ['RF', 'KNN', 'LR', 'MLP']
iterations = ['12', '11', '10', '9', '8', '7', '6', '5', '4', '3']
# colors = [
#         '#F0F8FF',
#         '#E1F5FE',
#         '#B3E5FC',
#         '#81D4FA',
#         '#4FC3F7',
#         '#29B6F6',
#         '#03A9F4',
#         '#0288D1',
#         '#FFCCBC',  ）
#         '#FF7043',
#         '#FF5722',
#         '#F44336',
#         '#D32F2F'
#     ]
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#bc80bd', '#80b1d3',
          '#fdb462', '#fb8072', '#F44336', '#D32F2F']
cmap = mcolors.ListedColormap(colors)


def plot_each_metric():
    balance_data_rev = balance_data[::-1]
    f1_data_rev = f1_data[::-1]
    auc_data_rev = auc_data[::-1]
    g_mean_data_rev = g_mean_data[::-1]
    datasets = [balance_data_rev, f1_data_rev, auc_data_rev, g_mean_data_rev]
    names = ['balance', 'F$_1$', 'AUC', 'G-mean']

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    for idx, (ax, values, metric) in enumerate(zip(axes.flat, datasets, names)):
        ax.imshow(values, cmap=cmap, aspect='auto', vmin=np.min(values), vmax=np.max(values))

        ax.set_xticks(np.arange(len(classifiers)))
        ax.set_yticks(np.arange(len(iterations)))
        ax.set_xticklabels(classifiers, fontfamily='Times New Roman', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_yticklabels(iterations, fontfamily='Times New Roman', fontsize=12, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

        for i in range(len(iterations)):
            for j in range(len(classifiers)):
                ax.text(j, i, f'{values[i, j]}',
                        ha="center", va="center", color="black",
                        fontsize=11, fontweight='bold', fontfamily='Times New Roman')
        ax.set_xticks(np.arange(-0.5, len(classifiers)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(iterations)), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        label_chr = chr(97 + idx)
        ax.set_xlabel(f'({label_chr}) {metric}',
                      fontsize=12,
                      fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_all():
    total_data = balance_data + f1_data + auc_data + g_mean_data
    total_data_rev = total_data[::-1]

    fig, ax = plt.subplots(figsize=(2.55, 3))
    ax.imshow(total_data_rev, cmap=cmap, aspect=0.5)
    ax.set_xticks(np.arange(len(classifiers)))
    ax.set_yticks(np.arange(len(iterations)))
    ax.set_xticklabels(classifiers, fontfamily='Times New Roman', fontsize=10, fontweight='bold')
    ax.set_yticklabels(iterations, fontfamily='Times New Roman', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='both', length=0)

    for i in range(len(iterations)):
        for j in range(len(classifiers)):
            value = total_data_rev[i, j]
            ax.text(j, i, f'{value}',
                    ha="center", va="center", color="black",
                    fontsize=9, fontweight='bold', fontfamily='Times New Roman')
    ax.set_xticks(np.arange(-0.5, len(classifiers)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(iterations)), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    ax.set_xlabel('classifier', fontsize=10, fontweight='bold',
                  fontfamily='Times New Roman', labelpad=6)
    ax.set_ylabel('iteration', fontsize=10, fontweight='bold',
                  fontfamily='Times New Roman', labelpad=6)
    plt.tight_layout()
    plt.subplots_adjust(left=0.16, right=1, bottom=0.13, top=0.98)
    plt.show()


# plot_each_metric()
plot_all()
