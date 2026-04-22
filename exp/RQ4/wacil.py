import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class WACIL:
    def fit_resample(self, X_train, y_train):
        colc = X_train.shape[1]
        rann = 3
        # borderline instance selecetion phase
        data = pd.DataFrame(data=X_train)
        data['defect'] = y_train
        unique_labels, counts = np.unique(y_train, return_counts=True)
        s = unique_labels[np.argmin(counts)]
        mask = data['defect'] == s
        Smin = data[mask]
        Smaj = data[~mask]
        Smin = Smin.values
        Smaj = Smaj.values
        XXmin = Smin[:, :-1]
        XXmaj = Smaj[:, :-1]
        k2 = 3
        min_len = len(Smin)
        maj_len = len(Smaj)
        TT = maj_len - min_len  # number of  pseudo instances to be generated
        k = 5
        k1 = k + 1
        data1 = np.vstack((XXmaj, XXmin))
        fstmp = np.zeros((0, colc))
        for i in range(len(XXmin)):
            sim_scores1 = XXmin.dot(XXmin[i]) / (np.linalg.norm(XXmin, axis=1) * np.linalg.norm(XXmin[i]))
            sim_scores2 = XXmaj.dot(XXmin[i]) / (np.linalg.norm(XXmaj, axis=1) * np.linalg.norm(XXmin[i]))
            dfs = np.sum(sim_scores1[sim_scores1.argsort()[-k1:][::-1][1:]])
            nfs = np.sum(sim_scores2[sim_scores2.argsort()[-k:][::-1][0:]])
            if (dfs >= nfs):
                fstmp = np.vstack((fstmp, XXmin[i]))
        nfstmp = np.zeros((0, colc))
        for i in range(len(fstmp)):
            sim_scores3 = X_train.dot(fstmp[i]) / (np.linalg.norm(X_train, axis=1) * np.linalg.norm(fstmp[i]))
            indc = sim_scores3.argsort()[-k1:][::-1][1:]
            cnt = np.count_nonzero(y_train[indc])
            if (cnt >= k2):
                indc2 = np.where(y_train[indc] == 1)
                nfstmp = np.vstack((nfstmp, X_train[indc[indc2[0]]]))
        nfstmp = np.unique(nfstmp, axis=0)
        fsbinc = np.zeros((0, colc))
        for i in range(len(nfstmp)):
            sim_scores4 = XXmin.dot(nfstmp[i]) / (np.linalg.norm(XXmin, axis=1) * np.linalg.norm(nfstmp[i]))
            ind = sim_scores4.argsort()[-k:][::-1][0:]
            fsbinc = np.vstack((fsbinc, XXmin[ind]))
        fsbinc = np.unique(fsbinc, axis=0)
        Xmin = fsbinc if len(fsbinc) >= rann else XXmin
        Xmin = pd.DataFrame(data=Xmin)
        b_len = len(Xmin)
        # pseudo-instance generation phase
        snum = 0
        while (snum < TT):
            mu = np.mean(Xmin)
            V = np.cov(Xmin.T)
            VI = np.linalg.pinv(V)
            list1 = np.diag(np.dot(np.dot((Xmin - mu), VI), (Xmin - mu).T))
            Xmin['mdist'] = list1
            Xmin = Xmin.sort_values(by='mdist')
            Xmin = Xmin.drop(columns=['mdist'])
            ott = np.zeros((rann, colc))
            Xmin1 = Xmin.values
            n1 = len(Xmin)
            rng = int(n1 / rann)
            for i in range(0, rng):
                ott[0] = Xmin1[i, 0:colc]
                ott[1] = Xmin1[i + rng, 0:colc]
                ott[2] = Xmin1[i + rng + rng, 0:colc]
                if (snum < TT):
                    ran = np.random.rand(colc)
                    min1 = ott.min(axis=0)
                    max1 = ott.max(axis=0)
                    tmp = min1 + ran * (max1 - min1)
                    tem = np.array([min1])
                    # filtering phase
                    tmp1 = np.array([tmp])
                    temp = cosine_similarity(tmp1, data1)
                    ind = (-temp[0]).argsort()[:k]
                    k_maj = len(list(j for j in ind if 0 <= j < maj_len))
                    k_min = k - k_maj
                    if (k_min >= k_maj and k_min != k):
                        df1 = pd.DataFrame(data=tmp1)
                        Xmin = Xmin._append(df1)
                        snum = snum + 1
                    else:
                        df2 = pd.DataFrame(data=tem)
                        Xmin = Xmin._append(df2)
                        snum += 1
                else:
                    break

        y_tr = np.ones(TT)
        Xminv = Xmin.values
        X_tr = Xminv[b_len:len(Xmin), :]
        X_train = np.vstack((X_train, X_tr))  # final training data
        y_train = np.concatenate((y_train, y_tr), axis=0)

        return X_train, y_train