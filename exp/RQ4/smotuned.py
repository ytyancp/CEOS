import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score


def parse_results(res):
    res = res.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    res = res.replace('array', '').replace(' ', '')
    res_to_list = res.split(",")
    res_to_list = [float(i) for i in res_to_list]
    res_to_list[0] = int(np.round_(res_to_list[0]))
    res_to_list[1] = int(np.round_(res_to_list[1]))
    res_to_list[2] = int(np.round_(res_to_list[2]))

    return res_to_list


def SMOTUNEDOversampling(X_train, X_test, y_train, y_test, model="DT"):
    if model == "RF":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=rf_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )
    elif model == "KNN":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=knn_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )
    elif model == "LR":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=lr_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )
    elif model == "DT":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=dt_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )
    elif model == "SVM":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=svm_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )
    elif model == "Adaboost":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=adaboost_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )
    elif model == "GBDT":
        smotuned_result = list(
            DESmote(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                func=gbdt_smotuned_func, bounds=[(50, 500), (1, 6), (5, 21)]
            )
        )

    result_to_string = str(smotuned_result[-1])
    result_to_list = parse_results(result_to_string)

    label = [y for y in y_train.values.tolist()]
    X_train_new, y_train_new = balance(X_train.values, label,
                                       m=result_to_list[0],
                                       r=result_to_list[1],
                                       neighbours=result_to_list[2])

    col = X_train.columns
    tar = y_train.name
    X_train_new = pd.DataFrame(X_train_new, columns=col)
    y_train_new = pd.Series(y_train_new, name=tar)

    return X_train_new, y_train_new


def DESmote(X_train, X_test, y_train, y_test, func, bounds, F=0.8, CR=0.9, pop_size=50, iter=10):
    dimensions = len(bounds)
    pop = np.random.rand(pop_size, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(int(np.round_(index[0])))
        temp_list.append(int(np.round_(index[1])))
        temp_list.append(int(np.round_(index[2])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(X_train, X_test, y_train, y_test,
                               index[0], index[1], index[2]) for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(iter):
        for j in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)

            for i, v in enumerate(mutant):
                if 0 < v < 1:
                    continue
                if v < 0:
                    mutant[i] = v + 1
                if v > 1:
                    mutant[i] = v - 1

            cross_points = np.random.rand(dimensions) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trial_denorm_convert = trial_denorm.tolist()
            f = func(X_train, X_test, y_train, y_test,
                     int(np.round_(trial_denorm_convert[0])),
                     int(np.round_(trial_denorm_convert[1])),
                     int(np.round_(trial_denorm_convert[2])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial

                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        yield best, fitness[best_idx]


def my_smote(data, num, k=3, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", p=r).fit(data)
    _, indices = nbrs.kneighbors(data)

    for _ in range(0, num):
        mid = random.randint(0, len(data) - 1)
        nn = indices[mid, random.randint(1, k - 1)]
        datamade = []

        for j in range(0, len(data[mid])):
            gap = random.random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])

        corpus.append(datamade)

    corpus = np.array(corpus)
    corpus = np.vstack((corpus, np.array(data)))

    return corpus


def balance(data_train, train_label, m, r, neighbours):
    pos_train = []
    neg_train = []
    unique_labels, counts = np.unique(train_label, return_counts=True)
    minority_label = unique_labels[np.argmin(counts)]
    for j, i in enumerate(train_label):
        if i == minority_label:
            pos_train.append(data_train[j])
        else:
            neg_train.append(data_train[j])

    pos_train = np.array(pos_train)
    neg_train = np.array(neg_train)

    if len(pos_train) < len(neg_train):
        pos_train = my_smote(pos_train, m, k=neighbours, r=r)

        if len(neg_train) >= len(pos_train):
            neg_train = neg_train[np.random.choice(len(neg_train), len(pos_train), replace=False)]
        else:
            pos_train = pos_train[np.random.choice(len(pos_train), len(neg_train), replace=False)]

    data_train1 = np.vstack((pos_train, neg_train))
    label_train = [1] * len(pos_train) + [0] * len(neg_train)

    return data_train1, label_train


def rf_smotuned_func(X_train, X_test, y_train, y_test,
                     m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_RF = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf_RF.fit(train_balanced_x, train_balanced_y)
    predictions = clf_RF.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def knn_smotuned_func(X_train, X_test, y_train, y_test,
                      m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_KNN = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf_KNN.fit(train_balanced_x, train_balanced_y)
    predictions = clf_KNN.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def lr_smotuned_func(X_train, X_test, y_train, y_test,
                     m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_LR = LogisticRegression(random_state=42, solver='saga', max_iter=20000, n_jobs=-1)
    clf_LR.fit(train_balanced_x, train_balanced_y)
    predictions = clf_LR.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def dt_smotuned_func(X_train, X_test, y_train, y_test,
                     m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_DT = DecisionTreeClassifier()
    clf_DT.fit(train_balanced_x, train_balanced_y)
    predictions = clf_DT.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def svm_smotuned_func(X_train, X_test, y_train, y_test,
                      m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_SVM = SVC()
    clf_SVM.fit(train_balanced_x, train_balanced_y)
    predictions = clf_SVM.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def adaboost_smotuned_func(X_train, X_test, y_train, y_test,
                           m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_Adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf_Adaboost.fit(train_balanced_x, train_balanced_y)
    predictions = clf_Adaboost.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def gbdt_smotuned_func(X_train, X_test, y_train, y_test,
                       m, r, neighbours):
    lab = [y for y in y_train.values.tolist()]
    train_balanced_x, train_balanced_y = balance(
        X_train.values,
        lab,
        m=m,
        r=r,
        neighbours=neighbours
    )

    col = X_train.columns
    tar = y_train.name
    train_balanced_x = pd.DataFrame(train_balanced_x, columns=col)
    train_balanced_y = pd.Series(train_balanced_y, name=tar)

    scaler = StandardScaler()
    train_balanced_x = pd.DataFrame(scaler.fit_transform(train_balanced_x), columns=col)
    X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=col)

    clf_GBDT = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    clf_GBDT.fit(train_balanced_x, train_balanced_y)
    predictions = clf_GBDT.predict(X_test_scale)

    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
    recall = recall_score(y_true=y_test, y_pred=predictions)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score
