# All the majority samples are used strategy is implemented in this file.
import numpy as np
import faiss
from joblib import Parallel, delayed


class FM:

    def __init__(self, k_neighbors=5, n_iterations=5):
        """
        Initialize the Counterfactual Explainable Oversampling (CEOS) algorithm.

        param k_neighbors: Number of nearest neighbors.
        param n_iterations: Number of iterations.

        FAISS（Facebook AI Similarity Search）is a library for efficient similarity search and dense vector clustering.
        For the CPU version, you can use the following command: pip install faiss-cpu
        For the GPU version, you can use the following command: pip install faiss-gpu
        """
        self.k_neighbors = k_neighbors
        self.n_iterations = n_iterations

        self.model = None
        self.minority_label = None
        self.majority_label = None

    def _data_separating(self, X, y):
        """
        Separate the dataset into clean(majority_label) and buggy(minority_label) classes.

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
        Train FAISS model, which uses L2 distance for similarity calculation.

        param X: Original feature dataset.
        :return: Trained faiss model.
        """
        d = X.shape[1]
        model = faiss.IndexFlatL2(d)  # build the index
        model.add(X.astype('float32'))  # add vectors to the index

        return model

    def _m2ms(self, X, y):
        """
        Switch the potential clean samples to buggy samples.

        param X: Feature dataset.
        param y: Class labels.
        :return: Feature dataset and updated labels.
        """
        clean_data, buggy_data = self._data_separating(X, y)

        y_new = y.copy()
        switched_indices = set()
        overlapping_samples = set()

        for clean_sample in clean_data:  # nearest neighbor to find samples of overlapping regions in 'clean'
            distances, indices = self.model.search(clean_sample.reshape(1, -1), self.k_neighbors + 1)  # remove itself
            labels = y[indices[0][1:]]
            minority_count = np.sum(labels == self.minority_label)
            if minority_count > self.k_neighbors // 2:
                overlapping_samples.add(tuple(clean_sample))

        for buggy_sample in buggy_data:
            distances, indices = self.model.search(buggy_sample.reshape(1, -1), self.k_neighbors + 1)
            for index in indices[0][1:]:
                nearest_label = y[index]
                # Update if neighbor class is 'clean' and this sample is in 'overlapping' and has not yet switched
                if nearest_label == self.majority_label and index not in switched_indices and tuple(
                        X[index]) in overlapping_samples:
                    y_new[index] = self.minority_label
                    switched_indices.add(index)

        return X, y_new

    def _process_cf_item(self, buggy_instance, clean_data, y):
        """
        Process counterfactual instances to obtain credible counterfactual instances.

        param buggy_instance: An instance from the buggy class.
        param clean_data: Data whose label is 'majority_label'.
        :return: List items. Such as [[(clean_index1, cf_sample1, score1)], [(clean_index2, cf_sample2, score2)],...]].
        """
        items = []
        for j in range(clean_data.shape[0]):
            cf_sample, score = self._mmos(clean_data[j], buggy_instance, y)
            if cf_sample is not None and score > 0:
                items.append((j, cf_sample, score))
        return items

    def _mmos(self, clean_instance, buggy_instance, y):
        """
        Generate a credible counterfactual instance via a clean instance and a buggy instance.

        param clean_instance: An instance selected from clean class.
        param buggy_instance: An instance selected from buggy class.
        param y: Class labels.
        :return: A credible counterfactual instance and its score.
        """
        left = 0
        right = 1
        score = 0
        credible_cf_instance = None
        is_buggy_generated = is_clean_generated = False  # track class status

        arrow_vector = buggy_instance - clean_instance

        for i in range(self.n_iterations):
            mid = (left + right) / 2
            current_instance = clean_instance + mid * arrow_vector

            indices = self.model.search(current_instance.reshape(1, -1), self.k_neighbors)[1]
            count = np.sum(y[indices[0]] == self.minority_label)
            if count > self.k_neighbors // 2:
                right = mid
                is_buggy_generated = True
                current_label = self.minority_label
            else:
                left = mid
                is_clean_generated = True
                current_label = self.majority_label

            # Score rules: 'clean': 1, 'buggy': -1, 'clean->buggy' or 'buggy->clean': 1
            if current_label == self.minority_label and is_clean_generated:
                is_clean_generated = False  # for real-time tracking of class status
                score += 2 * (i + 1)
                if credible_cf_instance is None:
                    credible_cf_instance = current_instance
                else:
                    credible_cf_instance = (credible_cf_instance + current_instance) / 2
            elif current_label == self.minority_label:
                score += 1 * (i + 1)
                if credible_cf_instance is not None:
                    credible_cf_instance = (credible_cf_instance + current_instance) / 2

            if current_label == self.majority_label and is_buggy_generated:
                is_buggy_generated = False  # 'score -= 0 * (i + 1)' is omitted here.
            elif current_label == self.majority_label:
                score -= 1 * (i + 1)

        return credible_cf_instance, score

    def fit(self, X, y):
        self.model = self._train_faiss_model(X)
        # Step 1: Majority-to-minority switching (M2MS)
        X_new, y_new = self._m2ms(X, y)
        clean_data, buggy_data = self._data_separating(X_new, y_new)

        if buggy_data.shape[0] >= clean_data.shape[0]:
            return X_new, y_new
        else:
            num_need = clean_data.shape[0] - buggy_data.shape[0]
        # Step 2: Majority-with-minority oversampling (MMOS)
        cf_item = Parallel(n_jobs=-1)(
            delayed(self._process_cf_item)(buggy_data[i], clean_data, y_new) for i in range(buggy_data.shape[0]))

        return X_new, y_new, cf_item, num_need

    def _cfss(self, X, y, cf_item, num_need):
        """
        Counterfactual adaptive merging in training data.

        param cf_item: List items like [[(clean_index1, cf_sample1, score1)], [(clean_index2, cf_sample2, score2)],..]].
        param num_need: Number of instances needed to be generated.
        :return: Balanced dataset and added counterfactual samples.
        """
        cf_item = {i: item for i, item in enumerate(cf_item) if item}
        # Remove the empty ones from dictionary and sort in descending order by score(x[2])
        processed_cf_item = {k: sorted(v, key=lambda x: x[2], reverse=True) for k, v in cf_item.items() if v}
        # Count the number of clean samples that match in buggy samples.
        total_count = sum(len(items) for items in processed_cf_item.values())
        # Assign the number of instances to be generated for each buggy sample.
        each_need_dict = {i: np.ceil(num_need * (len(items) / total_count)).astype(int) for i, items in
                          processed_cf_item.items()}
        X_synthetic = []
        y_synthetic = []
        for i, items in processed_cf_item.items():
            each_need = each_need_dict[i]
            if each_need == 0:
                continue
            for item in items[:each_need]:
                _, sample, _ = item
                X_synthetic.append(sample)
                y_synthetic.append(self.minority_label)

        if len(X_synthetic) == 0:
            return X, y, processed_cf_item
        X_balance = np.vstack((X, X_synthetic))
        y_balance = np.hstack((y, y_synthetic))

        return X_balance, y_balance, processed_cf_item

    def fit_resample(self, X, y):
        """
        Fit and resample the data.

        :return: Resampled dataset.
        """
        X_new, y_new, cf_item, num_need = self.fit(X, y)
        # step 3: Counterfactual adaptive merging
        X_balance, y_balance, _ = self._cfss(X_new, y_new, cf_item, num_need)

        return X_balance, y_balance
