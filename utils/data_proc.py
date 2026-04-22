import pandas as pd
from scipy.io import arff
import os
import matplotlib.pyplot as plt
import seaborn as sns


def arff_to_csv(path):
    arff_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".arff")]

    for arff_file in arff_files:
        data, meta = arff.loadarff(arff_file)
        df = pd.DataFrame(data)

        # Replace the class label with 0 and 1
        df['label'] = df['CLASS'].apply(lambda x: 1 if x.decode('utf-8') == 'MIN' else 0)
        df.drop('CLASS', axis=1, inplace=True)

        csv_file_name = os.path.splitext(arff_file)[0] + '.csv'
        df.to_csv(csv_file_name, index=False)


def dat_to_csv(path):
    dat_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".dat")]

    for dat_file in dat_files:
        df = pd.read_csv(dat_file, sep='\s+')
        csv_file = os.path.splitext(dat_file)[0] + '.csv'
        df.to_csv(csv_file, index=False)


def to_binary_data(path, output_path):
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)

            # df = df.drop(df.columns[-2], axis=1)  # drop the last two columns
            df.iloc[:, -1] = df.iloc[:, -1].map({1: 0, -1: 1})
            last_column_name = df.columns[-1]
            df[last_column_name] = df[last_column_name].apply(lambda x: 1 if x != 0 else 0)

            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, file)
            df.to_csv(save_path, index=False)


def plot_binary_data(path):
    df = pd.read_csv(path)

    plot_df = pd.DataFrame({
        'Feature 1': df.iloc[:, 0],
        'Feature 2': df.iloc[:, 1],
        'Label': df.iloc[:, -1].map({0: 'class 0', 1: 'class 1'})
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df,
                    x='Feature 1',
                    y='Feature 2',
                    hue='Label',
                    palette={'class 0': 'blue', 'class 1': 'red'},
                    style='Label',
                    s=100,
                    alpha=0.7)
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values
    #
    # plt.figure(figsize=(10, 6))
    #
    # plt.scatter(X[y == 0, 0], X[y == 0, 1],
    #             c='blue', label='class 0', alpha=0.6, edgecolors='w')
    # plt.scatter(X[y == 1, 0], X[y == 1, 1],
    #             c='red', label='class 1', alpha=0.6, edgecolors='w')

    plt.legend(loc='lower right')
    sns.despine()
    # plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


# arff_to_csv('path')  # your own dataset path
# dat_to_csv('path')
to_binary_data('path', 'output_path')
# plot_binary_data('path')
