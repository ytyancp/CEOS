# Only the oversampling stage is implemented in this file.
import numpy as np
import faiss
from joblib import Parallel, delayed


class OS:

    def __init__(self, k_neighbors=5, n_iterations=5):
        """
        Initialize the Counterfactual Explainable Oversampling (CEOS) algorithm.

        param k_neighbors: Number of nearest neighbors.
        param n_iterations: Number of iterations.

        FAISS（Facebook AI Similarity Search）is a library for efficient similarity search and dense vector clustering.
        For the CPU version, you can use the following command:
        pip install faiss-cpu
        For the GPU version, you can use the following command:
        pip install faiss-gpu
        """

        self.k_neighbors = k_neighbors
        self.n_iterations = n_iterations

        self.model = None
        self.minority_label = None
        self.majority_label = None

    def _data_separating(self, X, y):
        """
        Separate the dataset into majority and minority classes.

        param X: Feature dataset.
        param y: Class labels.
        :return: Clean and buggy class data.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        self.minority_label = unique_labels[np.argmin(counts)]
        self.majority_label = unique_labels[np.argmax(counts)]

        X_majority = X[np.where(y == self.majority_label)[0]]
        X_minority = X[np.where(y == self.minority_label)[0]]

        return X_majority, X_minority

    @staticmethod
    def _train_faiss_model(X):
        """
        Train the FAISS model, which uses L2 distance for similarity calculation.

        param X: Feature dataset.
        :return: Trained faiss model.
        """
        d = X.shape[1]
        model = faiss.IndexFlatL2(d)  # build the index
        model.add(X.astype('float32'))  # add vectors to the index

        return model

    def _process_buggy_item(self, buggy_instance, clean_data, y):
        """
        Process each buggy instance to generate counterfactual instances.

        param buggy_instance: An instance from the buggy class.
        param clean_data_cluster: Clean class data clustered.
        param y: Class labels.
        :return: List of generated counterfactual instances.
        """
        items = []

        for j in range(clean_data.shape[0]):
            cf_sample, score = self._mmos(clean_data[j], buggy_instance, y)
            if cf_sample is not None and score > 0:
                items.append((j, cf_sample, score))
        return items

    def _mmos(self, clean_instance, buggy_instance, y):
        """
        As minority samples are required, we use the majority ones to generate the counterfactuals (0->1)
        and omit the desire_label parameter (1->0).

        param clean_instance: An instance selected from clean class.
        param buggy_instance: An instance selected from buggy class.
        param y: Class labels.
        :return: A credible counterfactual instance.
        """
        left = 0
        right = 1
        score = 0
        credible_cf_instance = None
        is_buggy_generated = is_clean_generated = False

        arrow_vector = buggy_instance - clean_instance

        for i in range(self.n_iterations):

            mid = (left + right) / 2
            pseudo_instance = clean_instance + mid * arrow_vector

            distances, indices = self.model.search(pseudo_instance.reshape(1, -1), self.k_neighbors)
            neighbor_labels = y[indices[0]]
            count = np.sum(neighbor_labels == self.minority_label)

            if count >= self.k_neighbors // 2:
                right = mid
                is_buggy_generated = True
                speculative_current_label = self.minority_label
            else:
                left = mid
                is_clean_generated = True
                speculative_current_label = self.majority_label

            if speculative_current_label == self.minority_label and is_clean_generated:
                credible_cf_instance = pseudo_instance
                is_clean_generated = False
                score += 2 * (i + 1)
            elif speculative_current_label == self.minority_label:
                if credible_cf_instance is not None:
                    credible_cf_instance = (credible_cf_instance + pseudo_instance) / 2
                score += 1 * (i + 1)

            if speculative_current_label == self.majority_label and is_buggy_generated:
                is_buggy_generated = False
            elif speculative_current_label == self.majority_label:
                score -= 1 * (i + 1)

        return credible_cf_instance, score

    def _cfss(self, X, y, buggy_item, num_need):
        buggy_item = {i: item for i, item in enumerate(buggy_item) if item}
        processed_buggy_item = {k: sorted(v, key=lambda x: x[2], reverse=True) for k, v in buggy_item.items() if v}
        total_length = sum(len(items) for items in processed_buggy_item.values())
        each_need_dict = {i: np.ceil(num_need * (len(items) / total_length)).astype(int) for i, items in
                          processed_buggy_item.items()}
        X_synthetic = []
        y_synthetic = []
        for i, items in processed_buggy_item.items():
            each_need = each_need_dict[i]
            if each_need == 0:
                continue
            for item in items[:each_need]:
                j, sample, _ = item
                X_synthetic.append(sample)
                y_synthetic.append(self.minority_label)

        if len(X_synthetic) == 0:
            return X, y
        X_balance = np.vstack((X, X_synthetic))
        y_balance = np.hstack((y, y_synthetic))
        return X_balance, y_balance

    def fit_resample(self, X, y):
        self.model = self._train_faiss_model(X)

        clean_data, buggy_data = self._data_separating(X, y)

        if buggy_data.shape[0] >= clean_data.shape[0]:
            return X, y

        buggy_item = Parallel(n_jobs=-1)(
            delayed(self._process_buggy_item)(buggy_data[i], clean_data, y) for i in
            range(buggy_data.shape[0]))

        X_balance, y_balance = self._cfss(X, y, buggy_item, num_need=clean_data.shape[0] - buggy_data.shape[0])

        return X_balance, y_balance
