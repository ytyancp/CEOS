import pandas as pd
import numpy as np
import os
import multiprocessing
from utils.metrics import metric_sdp
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import xlsxwriter
import warnings
from ceos import CEOS
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


project = 'select'
classifier = 'KNN'
dataPath = f'../../dataset/{project}/'
listFile = os.listdir(dataPath)
first_row_name = ['DS', "CEOS"]

if classifier == 'MLP':
    sample_classifier = MLPClassifier()
elif classifier == 'RF':
    sample_classifier = RandomForestClassifier()
elif classifier == 'LR':
    sample_classifier = LogisticRegression()
elif classifier == 'KNN':
    sample_classifier = KNeighborsClassifier()


def five_fold_results(file_name, n):
    df = pd.read_csv(os.path.join(dataPath, file_name))
    data = df.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    ceos = CEOS(n_iterations=n)

    final_score = []
    for loop in range(10):
        scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(x, y):
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]

            x_balanced, y_balanced = ceos.fit_resample(x_train, y_train)
            sample_classifier.fit(x_balanced, y_balanced)
            y_pred = sample_classifier.predict(x_test)
            try:
                y_score = sample_classifier.predict_proba(x_test)[:, 1]
            except AttributeError:
                try:
                    y_score = sample_classifier.decision_function(x_test)
                except AttributeError:
                    raise ValueError("Can't calculate y_score, please check your classifier.")
            pf, balance = metric_sdp(y_test, y_pred)
            f_measure = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_score)
            g_mean = geometric_mean_score(y_test, y_pred)

            score = np.round([balance, f_measure, auc, g_mean], 4).tolist()
            scores.append(score)
        final_score.append(np.mean(scores, axis=0))
    result_dic = {'file': file_name, 'score': np.mean(final_score, axis=0)}
    return result_dic


def fit(file, n):
    final_result = []
    run_result = five_fold_results(file, n)
    final_result.append(run_result)
    print("\n")
    print("| File |Balance | F-Measure | AUC | G-Mean |")
    print("|------|--------|-----------|-----|-----|--------|")
    print(
        f"| {run_result['file']} |{run_result['score'][0]:.4f} | "
        f"{run_result['score'][1]:.4f} | {run_result['score'][2]:.4f} | {run_result['score'][3]:.4f} |")
    print("\n")
    print(f"{run_result['file']} is done.")
    return final_result


if __name__ == '__main__':

    n_iterations_list = list(range(3, 13))
    for n_iterations in n_iterations_list:
        savePath = f'../../results/para_{classifier}_{n_iterations}.csv'
        pool = multiprocessing.Pool(4)
        results = []
        for each_file in listFile:
            result = pool.apply_async(fit, args=[each_file, n_iterations])
            results.append(result)
        pool.close()
        pool.join()

        balance_result = []
        f_measure_result = []
        auc_result = []
        g_mean_result = []

        for res_dic in results:
            res = res_dic.get()
            each_file = res[0]['file']

            balance_temp_result = []
            f_measure_temp_result = []
            auc_temp_result = []
            g_mean_temp_result = []
            balance_temp_result.append(each_file)
            f_measure_temp_result.append(each_file)
            auc_temp_result.append(each_file)
            g_mean_temp_result.append(each_file)

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
