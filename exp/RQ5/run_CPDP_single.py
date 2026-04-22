import os
import warnings
import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils.metrics import metric_sdp
from ceos import CEOS
from NNfilter import NNFilter
from TCA import TCA
from BDA import BDA
from TCA_plus import TCA_plus
from exp.RQ4.STr_nn import STr_NN
from dpp import dpp
from exp.RQ4.mpos import MPOS

warnings.filterwarnings('ignore')
feature = 10
dataPath = '../../datasets/promise_cpdp'
listFile = os.listdir(dataPath)
save_path = f'../../results/single_results_{feature}.xlsx'


def select_most_similar_source(target_file, candidate_files):
    target_data = pd.read_csv(os.path.join(dataPath, target_file)).to_numpy()
    X_target = target_data[:, :-1]

    best_similarity = float('inf')
    best_source = None

    for source_file in candidate_files:
        source_data = pd.read_csv(os.path.join(dataPath, source_file)).to_numpy()
        X_source = source_data[:, :-1]

        feature_std = np.std(np.vstack([X_source, X_target]), axis=0)
        feature_std[feature_std == 0] = 1

        source_center = np.mean(X_source, axis=0)
        target_center = np.mean(X_target, axis=0)

        distance = cdist([source_center], [target_center],
                         metric='seuclidean', V=feature_std).flatten()[0]

        if distance < best_similarity:
            best_similarity = distance
            best_source = source_file

    return best_source


def get_class_centers(X_source, Y_source, X_target):
    unique_classes = np.unique(Y_source)
    n_features = X_source.shape[1]
    n_classes = len(unique_classes)

    source_class_centers = np.zeros((n_features, n_classes))
    target_distances = np.zeros((X_target.shape[0], n_classes))

    for i, class_label in enumerate(unique_classes):
        class_mask = (Y_source == class_label)
        X_class = X_source[class_mask]

        class_center = np.mean(X_class, axis=0)
        source_class_centers[:, i] = class_center

        feature_std = np.std(X_target, axis=0)
        feature_std[feature_std == 0] = 1

        distances = cdist(
            X_target.reshape(-1, n_features),
            class_center.reshape(1, -1),
            metric='seuclidean',
            V=feature_std
        ).flatten()

        target_distances[:, i] = distances

    return source_class_centers, target_distances


def dssdpp_predict(X_source, Y_source, X_target):
    if np.min(Y_source) > 0:
        Y_source_adjusted = Y_source - 1
    else:
        Y_source_adjusted = Y_source.copy()

    num_classes = len(np.unique(Y_source_adjusted))
    num_target_samples = X_target.shape[0]

    mpos = MPOS()
    X_balanced, Y_balanced = mpos.fit_resample(X_source, Y_source)

    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_balanced_selected = selector.fit_transform(X_balanced)
    X_target_selected = selector.transform(X_target)

    X_balanced_normalized = preprocessing.scale(X_balanced_selected)
    X_target_normalized = preprocessing.scale(X_target_selected)

    _, target_distances = get_class_centers(X_balanced_normalized, Y_balanced, X_target_normalized)
    membership_matrix = dpp(c=num_classes, nt=num_target_samples, d_ct_matrix=target_distances)

    if num_classes == 2:
        scores = membership_matrix[:, 1]
    else:
        scores = np.max(membership_matrix, axis=1)

    return scores


def str_nn_predict(X_source, Y_source, X_target, Y_target, feature_dim):
    X_source_norm = preprocessing.scale(X_source)
    X_target_norm = preprocessing.scale(X_target)

    tca = TCA(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
    Xs_new, Xt_new = tca.fit(X_source_norm, X_target_norm)
    str_nn = STr_NN(Xs_new, Y_source, Xt_new, Y_target, t=5)
    y_pred = str_nn.predict()

    return y_pred


def predict_with_classifiers(X_train, Y_train, X_test, Y_test, sample_weights=None):
    classifiers = {
        'RF': RandomForestClassifier(),
        'MLP': MLPClassifier(),
        'KNN': KNeighborsClassifier(),
        'LR': LogisticRegression()
    }

    results = {}

    for name, clf in classifiers.items():
        if name == 'LR' and sample_weights is not None:
            clf.fit(X_train, Y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, Y_train)

        y_pred = clf.predict(X_test)
        try:
            y_score = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            try:
                y_score = clf.decision_function(X_test)
            except AttributeError:
                raise ValueError("Can't calculate y_score, please check your classifier.")

        pf, balance = metric_sdp(Y_test, y_pred)
        f_measure = f1_score(Y_test, y_pred)
        auc = roc_auc_score(Y_test, y_score)
        g_mean = geometric_mean_score(Y_test, y_pred)

        results[name] = [balance, f_measure, auc, g_mean]

    return results


def calculate_metrics_from_predictions(y_true, y_pred):
    pf, balance = metric_sdp(y_true, y_pred)
    f_measure = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    g_mean = geometric_mean_score(y_true, y_pred)

    return [balance, f_measure, auc, g_mean]


def run_experiment(target_file, source_file, feature_dim):
    target_data = pd.read_csv(os.path.join(dataPath, target_file)).to_numpy()
    source_data = pd.read_csv(os.path.join(dataPath, source_file)).to_numpy()

    X_source, Y_source = source_data[:, :-1], source_data[:, -1]
    X_target, Y_target = target_data[:, :-1], target_data[:, -1]

    methods = ['TCA', 'NNFilter', 'TCA+', 'BDA', 'STr-NN', 'DSSDPP', 'TCA+CEOS', 'TCA++CEOS']
    results = {method: {} for method in methods}

    for method in methods:
        print(f"Running {method} for {target_file} with source {source_file}")

        if method == 'DSSDPP':
            scores = dssdpp_predict(X_source, Y_source, X_target)
            y_pred_dssdpp = (scores > 0.5).astype(int)
            metrics = calculate_metrics_from_predictions(Y_target, y_pred_dssdpp)

            for clf_name in ['RF', 'MLP', 'KNN', 'LR']:
                results[method][clf_name] = metrics

        elif method == 'STr-NN':
            y_pred_str_nn = str_nn_predict(X_source, Y_source, X_target, Y_target, feature_dim)
            metrics = calculate_metrics_from_predictions(Y_target, y_pred_str_nn)

            for clf_name in ['RF', 'MLP', 'KNN', 'LR']:
                results[method][clf_name] = metrics

        else:
            X_source_norm = preprocessing.scale(X_source)
            X_target_norm = preprocessing.scale(X_target)

            if method == 'NNFilter':
                nn_filter = NNFilter()
                X_train, Y_train = nn_filter.filter(10, X_source_norm, X_target_norm, Y_source)
                clf_results = predict_with_classifiers(X_train, Y_train, X_target_norm, Y_target)

            elif method == 'BDA':
                bda = BDA(kernel_type='linear', dim=feature_dim, lamb=1, mu=0.5, mode='BDA', gamma=1, estimate_mu=False)
                X_train, X_test = bda.fit(X_source_norm, Y_source, X_target_norm, Y_target)
                clf_results = predict_with_classifiers(X_train, Y_source, X_test, Y_target)

            elif method == 'TCA+':
                tca_plus = TCA_plus(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
                DCV_s = tca_plus.get_characteristic_vector(X_source)
                DCV_t = tca_plus.get_characteristic_vector(X_target)
                normalization_option = tca_plus.select_normalization_method(DCV_s, DCV_t)
                new_Xs, new_Xt = tca_plus.get_normalization_result(X_source, X_target, method_type=normalization_option)
                X_train, X_test = tca_plus.fit(new_Xs, new_Xt)
                clf_results = predict_with_classifiers(X_train, Y_source, X_test, Y_target)

            elif method == 'TCA':
                tca = TCA(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
                X_train, X_test = tca.fit(X_source_norm, X_target_norm)
                clf_results = predict_with_classifiers(X_train, Y_source, X_test, Y_target)

            elif method == 'TCA+CEOS':
                tca = TCA(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
                X_train, X_test = tca.fit(X_source_norm, X_target_norm)
                ceos = CEOS()
                X_resampled, y_resampled = ceos.fit_resample(X_train, Y_source)
                clf_results = predict_with_classifiers(X_resampled, y_resampled, X_test, Y_target)

            elif method == 'TCA++CEOS':
                tca_plus = TCA_plus(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
                DCV_s = tca_plus.get_characteristic_vector(X_source)
                DCV_t = tca_plus.get_characteristic_vector(X_target)
                normalization_option = tca_plus.select_normalization_method(DCV_s, DCV_t)
                new_Xs, new_Xt = tca_plus.get_normalization_result(X_source, X_target, method_type=normalization_option)
                X_train, X_test = tca_plus.fit(new_Xs, new_Xt)
                ceos = CEOS()
                X_resampled, y_resampled = ceos.fit_resample(X_train, Y_source)
                clf_results = predict_with_classifiers(X_resampled, y_resampled, X_test, Y_target)

            results[method] = clf_results

    return results


def save_results(all_results, save_path):
    metrics = ['balance', 'f_measure', 'auc', 'g_mean']
    classifiers = ['RF', 'MLP', 'KNN', 'LR']
    methods = ['TCA', 'NNFilter', 'TCA+', 'BDA', 'STr-NN', 'DSSDPP', 'TCA+CEOS', 'TCA++CEOS']

    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        for metric_idx, metric in enumerate(metrics):
            for clf in classifiers:
                data = []
                columns = ['Source', 'Target'] + methods

                for target_file, source_file, results in all_results:
                    row = [source_file, target_file]
                    for method in methods:
                        if method in ['DSSDPP', 'STr-NN']:
                            value = results[method]['RF'][metric_idx]
                        else:
                            value = results[method][clf][metric_idx] if clf in results[method] else 0
                        row.append(round(value, 4))
                    data.append(row)

                df = pd.DataFrame(data, columns=columns)
                sheet_name = f'{clf}_{metric}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == '__main__':
    all_results = []

    for target_file in tqdm(listFile, desc="Processing target projects"):
        candidate_files = [f for f in listFile if f != target_file]
        best_source_file = select_most_similar_source(target_file, candidate_files)

        print(f"Target: {target_file}, Selected Source: {best_source_file}")

        results = run_experiment(target_file, best_source_file, feature)
        all_results.append((target_file, best_source_file, results))

    save_results(all_results, save_path)
    print(f"All experiments completed! Results saved to {save_path}")
