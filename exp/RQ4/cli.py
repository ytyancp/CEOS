import copy
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from cleanlab.filter import find_label_issues
from cleanlab.count import compute_confident_joint, estimate_py_and_noise_matrices_from_probabilities
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class CLI:
    """
    CLI uses random under-sampling (RUS) to clean the training data,
    and then uses confident learning to remove data points that may have label errors.
    """

    def __init__(self, random_state=None, n_splits=5):

        self.random_state = random_state
        self.n_splits = n_splits
        self.rus = RandomUnderSampler(random_state=random_state)
        self.sample_weight = None

    def _get_probabilities(self, X_train, y_train):

        psx = np.zeros((len(y_train), 2))
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, y_train)):
            X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
            s_train_cv, s_holdout_cv = y_train[cv_train_idx], y_train[cv_holdout_idx]

            log_reg = LogisticRegression(solver='liblinear')
            log_reg.fit(X_train_cv, s_train_cv)
            psx_cv = log_reg.predict_proba(X_holdout_cv)
            psx[cv_holdout_idx] = psx_cv
        return psx

    def _clean_data(self, X_train, y_train, pre_new):
        """
        Using confident learning to clean the training data and remove data points that may have label errors.

        Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        pre_new (array-like): Predicted probabilities of the new data.

        Returns:
        tuple: X_train_clean, y_train_clean
        """
        X_train_copy = copy.deepcopy(X_train)
        y_train_copy = copy.deepcopy(y_train)
        pre_new_copy = copy.deepcopy(pre_new)

        y_train_int = y_train_copy.astype(np.int16)

        confident_joint = compute_confident_joint(
            labels=y_train_int,
            pred_probs=pre_new_copy,
            calibrate=False
        )

        py, noise_matrix, _, _ = estimate_py_and_noise_matrices_from_probabilities(
            labels=y_train_int, pred_probs=pre_new_copy)

        ordered_label_errors = find_label_issues(
            labels=y_train_int,
            pred_probs=pre_new_copy,
            confident_joint=confident_joint
        )

        x_mask = ~ordered_label_errors
        X_train_clean = X_train_copy[x_mask]
        y_train_clean = y_train_int[x_mask]

        self.sample_weight = np.ones(np.shape(y_train_clean))
        for k in range(2):
            sample_weight_k = 1.0 / noise_matrix[k][k]
            self.sample_weight[y_train_clean == k] = sample_weight_k

        return X_train_clean, y_train_clean

    def fit_resample(self, X_train, y_train):

        X_resampled, y_resampled = self.rus.fit_resample(X_train, y_train)

        psx = self._get_probabilities(X_resampled, y_resampled)

        X_clean, y_clean = self._clean_data(X_resampled, y_resampled, psx)

        return X_clean, y_clean
