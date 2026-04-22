import resample_methods
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
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

dataPath = "../../dataset/select"
listFile = os.listdir(dataPath)
savePath = '../../results/ablation_result.csv'
sample_classifier = RandomForestClassifier()

METHODS = ["Original", "CEOS_CS", "CEOS_OS", "CEOS_US"]
first_row_name = ['project', "Ori", "CEOS_CS", "CEOS_OS", "CEOS_US"]


def five_fold_results(file_name, method, sample_classifier):
    data = pd.read_csv(os.path.join(dataPath, file_name)).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    final_score = []
    for loop in range(10):
        scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            x_train, y_train = X[train_index], y[train_index]
            x_test, y_test = X[test_index], y[test_index]

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
    """
    :param file: current file
    :return: a list of five-fold results for each method
    """

    final_result = []
    for method in tqdm(METHODS):
        temp_result = five_fold_results(file, method, sample_classifier)
        final_result.append(temp_result)
    print("\n")
    print("| File |Balance | F-Measure | AUC | G-Mean |")
    print("|------|--------|-----------|-----|--------|")
    print(
        f"| {temp_result['file']} |{temp_result['score'][0]:.4f} | {temp_result['score'][1]:.4f} "
        f"| {temp_result['score'][2]:.4f} | {temp_result['score'][3]:.4f} |")
    print("\n")
    print(f"{temp_result['file']} is done.")
    return final_result


if __name__ == '__main__':
    pool = multiprocessing.Pool(4)

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
