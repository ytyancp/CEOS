import pandas as pd
import numpy as np
import os
import multiprocessing
from utils.metrics import metric_sdp
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import xlsxwriter
import warnings
from GAN.model import WGANGP
import resample_methods
from GAN.helper import get_cat_dims
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")

project = 'select'
classifier = 'KNN'
dataPath = f'../../dataset/{project}/'
listFile = os.listdir(dataPath)
savePath = f'../../results/final_{classifier}.csv'

if classifier == 'MLP':
    sample_classifier = MLPClassifier()
elif classifier == 'RF':
    sample_classifier = RandomForestClassifier()
elif classifier == 'LR':
    sample_classifier = LogisticRegression()
elif classifier == 'KNN':
    sample_classifier = KNeighborsClassifier()

METHODS = ["Original", "SMOTE", "BorderSMOTE", "MAHAKIL", "WACIL", "CFOS", "SBGAN", "WGANGP",
           "CBR", "MPOS", "MOSIG", "CFSVM", "CEOS"]
first_row_name = ['DS', "Ori", "SMOTE", "BSMOTE", "MAHAKIL", "WACIL", "CFOS", "SBGAN", "WGANGP",
                  "CBR",  "MPOS", "MOSIG", "CFSVM", "CEOS"]


def five_fold_results(file_name, method, sample_classifier):
    df = pd.read_csv(os.path.join(dataPath, file_name))
    data = df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    final_score = []
    for loop in range(5):
        scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            x_train, y_train = X[train_index], y[train_index]
            x_test, y_test = X[test_index], y[test_index]

            if method == "WGANGP":
                num_cols = list(df.columns[:-1])
                cat_cols = None
                cat_dims = get_cat_dims(x_train, cat_cols)
                gan = WGANGP(write_to_disk=False,
                             # whether to create an output folder. Plotting will be surpressed if flase
                             compute_metrics_every=1250, print_every=2500, plot_every=10000,
                             num_cols=num_cols, cat_dims=cat_dims,
                             # pass the one hot encoder to the GAN to enable count plots of categorical variables
                             transformer=None,
                             # pass column names to enable
                             cat_cols=cat_cols,
                             use_aux_classifier_loss=True,
                             d_updates_per_g=3, gp_weight=15)
                gan.fit(x_train, y=y_train,
                        condition=True,
                        epochs=100,
                        batch_size=64,
                        netG_kwargs={'hidden_layer_sizes': (128, 64),
                                     'n_cross_layers': 1,
                                     'cat_activation': 'gumbel_softmax',
                                     'num_activation': 'none',
                                     'condition_num_on_cat': False,
                                     'noise_dim': 30,
                                     'normal_noise': False,
                                     'activation': 'leaky_relu',
                                     'reduce_cat_dim': True,
                                     'use_num_hidden_layer': True,
                                     'layer_norm': False, },
                        netD_kwargs={'hidden_layer_sizes': (128, 64, 32),
                                     'n_cross_layers': 2,
                                     'embedding_dims': None,
                                     'activation': 'leaky_relu',
                                     'sigmoid_activation': False,
                                     'noisy_num_cols': True,
                                     'layer_norm': True, }
                        )
                x_train, y_train = gan.resample(x_train, y=y_train)
                model = sample_classifier
                model.fit(x_train, y_train)
            else:
                model = resample_methods.ResampleClassifier(base_estimator=sample_classifier)
                model.fit(x_train, y_train, method)
            y_pred = model.predict(x_test)
            try:
                y_score = sample_classifier.predict_proba(x_test)[:, 1]
            except AttributeError:
                try:
                    y_score = sample_classifier.decision_function(x_test)
                except AttributeError:
                    raise ValueError("Can't calculate y_score, please check your classifier.")
            pf, balance = metric_sdp(y_test, y_pred)
            # recall = recall_score(y_test, y_pred)
            f_measure = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_score)
            g_mean = geometric_mean_score(y_test, y_pred)

            score = np.round([balance, f_measure, auc, g_mean], 4).tolist()
            scores.append(score)
        final_score.append(np.mean(scores, axis=0))
    result = {'file': file_name, 'score': np.mean(final_score, axis=0)}
    return result


def fit(file):

    final_result = []
    for method in tqdm(METHODS):
        temp_result = five_fold_results(file, method, sample_classifier)
        final_result.append(temp_result)
    print("\n")
    print("| File |Balance | F-Measure | AUC | G-Mean |")
    print("|------|--------|-----------|-----|-----|--------|")
    print(
        f"| {temp_result['file']} |{temp_result['score'][0]:.4f} | {temp_result['score'][1]:.4f} "
        f"| {temp_result['score'][2]:.4f} | {temp_result['score'][3]:.4f} |")
    print("\n")
    print(f"{temp_result['file']} is done.")
    return final_result


if __name__ == '__main__':
    pool = multiprocessing.Pool(3)
    results = []
    for file in listFile:
        result = pool.apply_async(fit, args=[file])
        results.append(result)
    pool.close()
    pool.join()

    balance_result = []
    f_measure_result = []
    auc_result = []
    g_mean_result = []

    for res_dic in results:
        res = res_dic.get()
        file = res[0]['file']

        balance_temp_result = []
        f_measure_temp_result = []
        auc_temp_result = []
        g_mean_temp_result = []

        balance_temp_result.append(file)
        f_measure_temp_result.append(file)
        auc_temp_result.append(file)
        g_mean_temp_result.append(file)

        for temp_result in res:
            all_metrics = temp_result['score']
            balance_temp_result.append(all_metrics[0])
            f_measure_temp_result.append(all_metrics[1])
            auc_temp_result.append(all_metrics[2])
            g_mean_temp_result.append(all_metrics[3])

        balance_result.append(balance_temp_result)
        f_measure_result.append(f_measure_temp_result)
        auc_result.append(auc_temp_result)
        g_mean_result.append(g_mean_temp_result)

    balance_result = np.vstack(([np.array(first_row_name)], np.array(balance_result)))
    f_measure_result = np.vstack(([np.array(first_row_name)], np.array(f_measure_result)))
    auc_result = np.vstack(([np.array(first_row_name)], np.array(auc_result)))
    g_mean_result = np.vstack(([np.array(first_row_name)], np.array(g_mean_result)))

    wb = xlsxwriter.Workbook(savePath, {'constant_memory': True, 'strings_to_numbers': True})

    worksheets = ['balance', 'f_measure', 'auc', 'g_mean']
    results = [balance_result, f_measure_result, auc_result, g_mean_result]

    for sheet_name, result in zip(worksheets, results):
        worksheet = wb.add_worksheet(sheet_name)
        for row, item in enumerate(result):
            for column, value in enumerate(item):
                worksheet.write(row, column, value)
    wb.close()
