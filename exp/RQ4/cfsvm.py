import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.svm import SVC
from RBFModel import RBFKernelNumpy
from sklearn.model_selection import GridSearchCV
import scipy.optimize as opt


class CFSVM:
    def __init__(self, clf, num_iter=100, beta=10.0, X_train=None, n_clusters=-1, cluster_method='kmedoids', eps=1e-3):
        self.clf = clf
        self.num_iter = num_iter
        self.beta = beta
        self.C = C = clf.best_estimator_.C
        self.eps = eps
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters

        if isinstance(clf, SVC):
            self.xis = clf.support_vectors_
            self.alphas = abs(clf.dual_coef_[0])
            self.yis = (clf.dual_coef_[0] > 0) * 2 - 1
            self.b = clf.intercept_[0]
            self.gamma = clf.gamma if isinstance(clf.gamma, float) else clf._gamma
        if isinstance(clf, GridSearchCV):
            self.xis = clf.best_estimator_.support_vectors_
            self.alphas = abs(clf.best_estimator_.dual_coef_[0])
            self.yis = (clf.best_estimator_.dual_coef_[0] > 0) * 2 - 1
            self.b = clf.best_estimator_.intercept_[0]
            self.gamma = clf.best_estimator_.gamma if isinstance(clf.best_estimator_.gamma, float) else clf.best_estimator_._gamma
        else:
            raise ValueError('Unknown classifier type')

        self.kernel = RBFKernelNumpy(self.gamma)
        self.X_train = X_train

        if X_train is not None:
            self.existing_counterfactuals = self.xis[self.alphas < C]
            if n_clusters > 0:
                data = np.vstack([self.X_train, self.existing_counterfactuals])
                n_clusters = min(n_clusters, len(data))
                self.cluster = KMedoids(n_clusters=n_clusters, method='pam', random_state=42)
                self.cluster.fit(data)
            else:
                self.cluster = None

    def decision_function(self, x):
        if len(x.shape) > 1 and x.shape[0] > 10:
            return self.long_decision_function(x)
        return np.sum(self.alphas[:, None] * self.yis[:, None] * np.exp(
            -self.gamma * np.sum((x - self.xis[:, None]) ** 2, axis=2)), axis=0) + self.b

    def long_decision_function(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.decision_function(x))
        return np.array(y_pred).flatten()

    def _line_search(self, x, x_c, y_target):
        target_val = 1.0 if y_target == 1 else 0

        def objective(l):
            candidate = l * x + (1 - l) * x_c
            df_val = self.decision_function(candidate)
            return l ** 2 + self.beta * (df_val - target_val) ** 2

        result = opt.minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'maxiter': self.num_iter})
        return result.x * x + (1 - result.x) * x_c

    def _compute_counterfactual_candidates(self, x, y_target):
        if self.cluster is not None:
            x_c_list = self.cluster.cluster_centers_
        else:
            x_c_list = self.existing_counterfactuals

        if len(x_c_list) == 0:
            return []

        candidates = np.unique([self._line_search(x, x_c, y_target) for x_c in x_c_list], axis=0)
        return candidates

    def _is_valid_counterfactual(self, candidate, target):
        df_val = self.decision_function(candidate)
        if target == 1:
            return df_val >= 1 - self.eps
        else:
            return df_val <= -1 + self.eps

    def _compute_counterfactual(self, query_instance, target, N=1, strategy='greedy'):
        candidates = self._compute_counterfactual_candidates(query_instance, target)
        valid_candidates = [c for c in candidates if self._is_valid_counterfactual(c, target)]

        if N == 1:
            return valid_candidates[0] if valid_candidates else None

        if strategy == 'greedy':
            return sorted(valid_candidates, key=lambda x: np.linalg.norm(x - query_instance))[:N]
        elif strategy == 'random':
            np.random.shuffle(valid_candidates)
            return valid_candidates[:N] if len(valid_candidates) >= N else valid_candidates
        elif strategy == 'optimal':
            from itertools import combinations
            if len(valid_candidates) < N:
                return valid_candidates
            best_comb = max(combinations(valid_candidates, N), key=lambda comb: np.mean([np.linalg.norm(a - b) for a, b in zip(comb, comb[1:])]))
            return list(best_comb)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def fit_resample(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise ValueError("Only binary classification is supported")

        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        samples_needed = counts[np.argmax(counts)] - counts[np.argmin(counts)]

        if samples_needed <= 0:
            return X.copy(), y.copy()

        majority_mask = (y == majority_class)
        query_instances = X[majority_mask]

        counterfactuals = []
        for query in query_instances:
            if len(counterfactuals) >= samples_needed:
                break

            cfs = self._compute_counterfactual(query, minority_class, N=1)
            if cfs is not None:
                counterfactuals.append(cfs)

        if counterfactuals:
            counterfactuals = np.unique(counterfactuals, axis=0)
            cf_labels = np.full(len(counterfactuals), minority_class)
            X_resampled = np.vstack([X, counterfactuals])
            y_resampled = np.hstack([y, cf_labels])
            return X_resampled, y_resampled
        else:
            print("No counterfactuals found")
            return X.copy(), y.copy()
