import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9

methods = ['CLI', 'CFSVM', 'MOSIG', 'MPOS', 'STr-NN', 'CBR', 'WGANGP', 'SBGAN', 'CFOS',
           'WACIL', 'S-TUN', 'MAHAK', 'BSMO', 'SMO', 'Ori']
classifiers = ['RF', 'KNN', 'LR', 'MLP']
metrics = ['balance', 'f_measure', 'auc', 'g_mean']
metric_names = {'balance': 'balance', 'f_measure': 'F$_1$', 'auc': 'AUC', 'g_mean': 'G-mean'}


def load_statistical_data():
    data_dict = {}
    path = '../../results'

    for clf in classifiers:
        data_dict[clf] = {}
        file_path = f'{path}\\final_{clf}_statistics.xlsx'

        for metr in metrics:
            df = pd.read_excel(file_path, sheet_name=metr)

            p_value_row = df.iloc[-2]  # p-value
            cliff_row = df.iloc[-1]  # cliff delta
            p_values = []
            cliff_deltas = []

            for METHOD in methods:
                p_val = float(p_value_row[METHOD])
                cliff_val = float(cliff_row[METHOD])
                p_values.append(p_val)
                cliff_deltas.append(cliff_val)
            data_dict[clf][metric_names[metr]] = {
                'p': p_values,
                'δ': cliff_deltas
            }

    return data_dict


fig, ax = plt.subplots(figsize=(14, 6))

ax.set_xlim(-1, len(classifiers) * len(metrics) - 0.2)
ax.set_ylim(-0.5, len(methods) - 0.3)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

rect = plt.Rectangle((-0.5, -0.5), len(classifiers) * len(metrics), len(methods),
                     fill=False, edgecolor='black', linewidth=1.5)
ax.add_patch(rect)

for i, method in enumerate(methods):
    ax.text(-0.55, i, method, ha='right', va='center', fontsize=10, fontweight='bold')

for i in range(1, len(classifiers)):
    x_pos = i * len(metrics) - 0.5
    ax.plot([x_pos, x_pos], [-0.5, len(methods) - 0.5], color='black', linewidth=1.2)

for i, classifier in enumerate(classifiers):
    classifier_x = i * len(metrics) + (len(metrics) - 1) / 2
    ax.text(classifier_x, len(methods) - 0.3, classifier, ha='center', va='bottom',
            fontsize=10, fontweight='bold')

for i, classifier in enumerate(classifiers):
    for j, metric in enumerate(metrics):
        x_pos = i * len(metrics) + j + 0.05
        ax.text(x_pos, -0.55, metric_names[metric], ha='center', va='top',
                fontsize=9, fontweight='bold')


def get_p_value_level(p_value):
    if p_value > 0.05:
        return 'Insignificant', '#1f77b4'
    elif p_value > 0.01:
        return 'Significant', '#ff7f0e'
    elif p_value > 0.001:
        return 'Very Significant', '#ff4500'
    else:
        return 'Extremely Significant', '#d62728'


def get_effect_size_level(effect_size):
    abs_effect = abs(effect_size)
    if abs_effect < 0.147:
        return 'Negligible', '#1f77b4'
    elif abs_effect < 0.33:
        return 'Small', '#ff7f0e'
    elif abs_effect < 0.474:
        return 'Medium', '#ff4500'
    else:
        return 'Large', '#d62728'


def plot_statistical_results(test_data):
    marker_spacing = 0.2

    for idx, METHOD in enumerate(methods):
        for J, CLASSIFIER in enumerate(classifiers):
            for k, METRIC in enumerate(metrics):
                _x_pos = J * len(metrics) + k + 0.02
                display_metric = metric_names[METRIC]

                p_value = test_data[CLASSIFIER][display_metric]['p'][idx]
                effect_size = test_data[CLASSIFIER][display_metric]['δ'][idx]

                p_level, p_color = get_p_value_level(p_value)
                ax.scatter(_x_pos - marker_spacing / 2, idx, s=55, marker='o',
                           color=p_color, alpha=0.9, edgecolors='black', linewidth=0.5)

                es_level, es_color = get_effect_size_level(effect_size)
                ax.scatter(_x_pos + marker_spacing / 2, idx, s=55, marker='D',
                           color=es_color, alpha=0.9, edgecolors='black', linewidth=0.5)


def create_legend():
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                   markersize=7, label='p > 0.05'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#1f77b4',
                   markersize=7, label='|δ| < 0.147'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                   markersize=7, label='0.01 < p ≤ 0.05'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff7f0e',
                   markersize=7, label='0.147 ≤ |δ| < 0.33'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4500',
                   markersize=7, label='0.001 < p ≤ 0.01'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff4500',
                   markersize=7, label='0.33 ≤ |δ| < 0.474'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                   markersize=7, label='p ≤ 0.001'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728',
                   markersize=7, label='|δ| ≥ 0.474')
    ]

    legend = ax.legend(handles=legend_elements,
                       loc='upper center', bbox_to_anchor=(0.5, -0.04),
                       ncol=4, fontsize=9, frameon=False, columnspacing=1.5, handletextpad=0.3)

    for text in legend.get_texts():
        text.set_fontweight('bold')


data = load_statistical_data()
plot_statistical_results(data)
create_legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.11, top=0.96, left=0.02, right=1.014)

plt.show()
