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
from joblib import Parallel, delayed
import gc
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
dataPath = '../../dataset/promise_cpdp'
save_path = f'../../results/multi_results_{feature}.xlsx'


class CachedClassifier:

    def __init__(self, base_classifier, classifier_name):
        self.clf = base_classifier
        self.name = classifier_name
        self.cache = {}

    def fit_predict_metrics(self, X_train, y_train, X_test, y_test, cache_key):
        if cache_key in self.cache:
            return self.cache[cache_key]

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        try:
            y_score = self.clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            try:
                y_score = self.clf.decision_function(X_test)
            except AttributeError:
                raise ValueError("Can't calculate y_score, please check your classifier.")

        pf, balance = metric_sdp(y_test, y_pred)
        f_measure = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        g_mean = geometric_mean_score(y_test, y_pred)

        metrics = [balance, f_measure, auc, g_mean]

        self.cache[cache_key] = metrics
        return metrics


def preprocess_data(path, files):
    processed_data = {}

    for file in files:
        data = pd.read_csv(os.path.join(path, file)).to_numpy()
        X, y = data[:, :-1], data[:, -1]
        X_norm = preprocessing.scale(X)
        processed_data[file] = (X_norm, y)

    return processed_data


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

        from scipy.spatial.distance import cdist
        distances = cdist(X_target, class_center.reshape(1, -1),
                          metric='seuclidean', V=feature_std).flatten()
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

    str_nn = STr_NN(Xs_new, Y_source, Xt_new, Y_target, t=10)
    y_pred = str_nn.predict()

    return y_pred


def initialize_cached_classifiers():
    classifiers_config = {
        'RF': RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        'MLP': MLPClassifier(
            random_state=42,
            max_iter=500
        ),
        'KNN': KNeighborsClassifier(
            n_jobs=1
        ),
        'LR': LogisticRegression(
            random_state=42,
            n_jobs=1
        )
    }

    cached_classifiers = {}
    for name, clf in classifiers_config.items():
        cached_classifiers[name] = CachedClassifier(clf, name)

    return cached_classifiers


def calculate_metrics_from_predictions(y_true, y_pred):

    pf, balance = metric_sdp(y_true, y_pred)
    f_measure = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    g_mean = geometric_mean_score(y_true, y_pred)

    return [balance, f_measure, auc, g_mean]


def run_method(method, X_source, Y_source, X_target, Y_target, feature_dim, cached_classifiers, target_file):
    results = {}

    if method == 'DSSDPP':
        scores = dssdpp_predict(X_source, Y_source, X_target)
        y_pred_dssdpp = (scores > 0.5).astype(int)
        metrics = calculate_metrics_from_predictions(Y_target, y_pred_dssdpp)

        for clf_name in cached_classifiers.keys():
            results[clf_name] = metrics

    elif method == 'STr-NN':
        y_pred_str_nn = str_nn_predict(X_source, Y_source, X_target, Y_target, feature_dim)
        metrics = calculate_metrics_from_predictions(Y_target, y_pred_str_nn)

        for clf_name in cached_classifiers.keys():
            results[clf_name] = metrics

    else:
        X_source_norm = preprocessing.scale(X_source)
        X_target_norm = preprocessing.scale(X_target)
        X_train, Y_train, X_test = None, None, X_target_norm

        if method == 'NNFilter':
            nn_filter = NNFilter()
            X_train, Y_train = nn_filter.filter(10, X_source_norm, X_target_norm, Y_source)
            X_test = X_target_norm

        elif method == 'BDA':
            bda = BDA(kernel_type='linear', dim=feature_dim, lamb=1, mu=0.5,
                      mode='BDA', gamma=1, estimate_mu=False)
            X_train, X_test = bda.fit(X_source_norm, Y_source, X_target_norm, Y_target)
            Y_train = Y_source

        elif method == 'TCA+':
            tca_plus = TCA_plus(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
            DCV_s = tca_plus.get_characteristic_vector(X_source)
            DCV_t = tca_plus.get_characteristic_vector(X_target)
            normalization_option = tca_plus.select_normalization_method(DCV_s, DCV_t)
            new_Xs, new_Xt = tca_plus.get_normalization_result(X_source, X_target,
                                                               method_type=normalization_option)
            X_train, X_test = tca_plus.fit(new_Xs, new_Xt)
            Y_train = Y_source

        elif method == 'TCA':
            tca = TCA(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
            X_train, X_test = tca.fit(X_source_norm, X_target_norm)
            Y_train = Y_source

        elif method == 'TCA+CEOS':
            tca = TCA(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
            X_train, X_test = tca.fit(X_source_norm, X_target_norm)
            ceos = CEOS()
            X_resampled, y_resampled = ceos.fit_resample(X_train, Y_source)
            X_train, Y_train = X_resampled, y_resampled

        elif method == 'TCA++CEOS':
            tca_plus = TCA_plus(kernel_type='linear', dim=feature_dim, lamb=1, gamma=1)
            DCV_s = tca_plus.get_characteristic_vector(X_source)
            DCV_t = tca_plus.get_characteristic_vector(X_target)
            normalization_option = tca_plus.select_normalization_method(DCV_s, DCV_t)
            new_Xs, new_Xt = tca_plus.get_normalization_result(X_source, X_target,
                                                               method_type=normalization_option)
            X_train, X_test = tca_plus.fit(new_Xs, new_Xt)
            ceos = CEOS()
            X_resampled, y_resampled = ceos.fit_resample(X_train, Y_source)
            X_train, Y_train = X_resampled, y_resampled

        for clf_name, classifier in cached_classifiers.items():
            cache_key = f"{method}_{target_file}_{clf_name}"
            metrics = classifier.fit_predict_metrics(X_train, Y_train, X_test, Y_target, cache_key)
            results[clf_name] = metrics

    return results


def run_experiment(target_file, data, feature_dim):
    print(f"Start processing target file: {target_file}")

    X_target, Y_target = data[target_file]

    source_files = [f for f in listFile if f != target_file]

    X_source_list = []
    Y_source_list = []

    for source_file in source_files:
        X_source, Y_source = all_data[source_file]
        X_source_list.append(X_source)
        Y_source_list.append(Y_source)

    X_source = np.vstack(X_source_list)
    Y_source = np.hstack(Y_source_list)

    cached_classifiers = initialize_cached_classifiers()

    methods = ['TCA', 'NNFilter', 'TCA+', 'BDA', 'STr-NN', 'DSSDPP', 'TCA+CEOS', 'TCA++CEOS']
    results = {}

    for method in methods:
        results[method] = run_method(
            method, X_source, Y_source, X_target, Y_target,
            feature_dim, cached_classifiers, target_file
        )

    del X_source, Y_source, cached_classifiers
    gc.collect()

    print(f"Finish processing target file: {target_file}")
    return target_file, results


def save_results(results, path):
    metrics = ['balance', 'f_measure', 'auc', 'g_mean']
    classifiers = ['RF', 'MLP', 'KNN', 'LR']
    methods = ['TCA', 'NNFilter', 'TCA+', 'BDA', 'STr-NN', 'DSSDPP', 'TCA+CEOS', 'TCA++CEOS']

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        for metric_idx, metric in enumerate(metrics):
            for clf in classifiers:
                data = []
                columns = ['DS'] + methods
                for target_name, final_results in results:
                    row = [target_name]
                    for method in methods:
                        value = final_results[method][clf][metric_idx] if clf in final_results[method] else 0
                        row.append(round(value, 4))
                    data.append(row)

                df = pd.DataFrame(data, columns=columns)
                sheet_name = f'{clf}_{metric}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == '__main__':
    listFile = os.listdir(dataPath)
    all_data = preprocess_data(dataPath, listFile)

    all_results = Parallel(n_jobs=4)(
        delayed(run_experiment)(target_file, all_data, feature)
        for target_file in tqdm(listFile)
    )

    save_results(all_results, save_path)
    print(f"All experiments completed! Results saved to {save_path}")
