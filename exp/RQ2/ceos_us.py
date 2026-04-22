# We use undersampling strategy instead of switching the overlapping majority samples.
import numpy as np
import faiss
from joblib import Parallel, delayed


class US:

    def __init__(self, k_neighbors=5, n_iterations=5):
        """
        Initialize the Counterfactual Explainable Oversampling (CEOS) algorithm.
        """
        self.k_neighbors = k_neighbors
        self.n_iterations = n_iterations

        self.model = None
        self.minority_label = None
        self.majority_label = None
        self.original_X = None
        self.original_y = None

    def _data_separating(self, X, y):
        """
        Separate the dataset into clean(majority_label) and buggy(minority_label) classes.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        self.minority_label = unique_labels[np.argmin(counts)]
        self.majority_label = unique_labels[np.argmax(counts)]

        X_majority = X[np.where(y == self.majority_label)[0]]
        X_minority = X[np.where(y == self.minority_label)[0]]

        return X_majority, X_minority

    def _train_faiss_model(self, X):
        """
        Train FAISS model, which uses L2 distance for similarity calculation.
        """
        d = X.shape[1]
        model = faiss.IndexFlatL2(d)  # build the index
        model.add(X.astype('float32'))  # add vectors to the index
        return model

    def _m2ms(self, X, y):
        """
        Delete the potential overlapping majority samples instead of switching their labels.
        """
        clean_data, buggy_data = self._data_separating(X, y)

        # Find overlapping regions in 'clean' class using original indices
        overlapping_indices = set()
        original_clean_indices = np.where(y == self.majority_label)[0]

        for i, clean_sample in enumerate(clean_data):
            distances, indices = self.model.search(clean_sample.reshape(1, -1), self.k_neighbors + 1)

            # Map FAISS indices back to original dataset indices
            original_neighbor_indices = []
            for idx in indices[0][1:]:  # exclude the sample itself
                if idx < len(self.original_y):  # 确保索引在有效范围内
                    original_neighbor_indices.append(idx)

            if not original_neighbor_indices:
                continue

            labels = self.original_y[original_neighbor_indices]
            minority_count = np.sum(labels == self.minority_label)

            if minority_count > self.k_neighbors // 2:
                original_index = original_clean_indices[i]
                overlapping_indices.add(original_index)

        # Create masks for samples to keep
        keep_mask = np.ones(len(y), dtype=bool)
        for idx in overlapping_indices:
            if idx < len(keep_mask):
                keep_mask[idx] = False

        # Apply deletion
        X_new = X[keep_mask]
        y_new = y[keep_mask]

        return X_new, y_new

    def _process_cf_item(self, buggy_instance, clean_data, y):
        """
        Process counterfactual instances to obtain credible counterfactual instances.
        """
        items = []
        for j in range(clean_data.shape[0]):
            cf_sample, score = self._oversampling(clean_data[j], buggy_instance, y)
            if cf_sample is not None and score > 0:
                items.append((j, cf_sample, score))
        return items

    def _oversampling(self, clean_instance, buggy_instance, y):
        """
        Generate a credible counterfactual instance via a clean instance and a buggy instance.
        """
        left = 0
        right = 1
        score = 0
        credible_cf_instance = None
        is_buggy_generated = is_clean_generated = False

        arrow_vector = buggy_instance - clean_instance

        for i in range(self.n_iterations):
            mid = (left + right) / 2
            current_instance = clean_instance + mid * arrow_vector

            # Search in the ORIGINAL dataset for neighborhood information
            indices = self.model.search(current_instance.reshape(1, -1).astype('float32'), self.k_neighbors)[1]

            # Ensure indices are within bounds of original_y
            valid_indices = [idx for idx in indices[0] if idx < len(self.original_y)]
            if not valid_indices:
                count = 0
            else:
                labels = self.original_y[valid_indices]
                count = np.sum(labels == self.minority_label)

            if count > self.k_neighbors // 2:
                right = mid
                is_buggy_generated = True
                current_label = self.minority_label
            else:
                left = mid
                is_clean_generated = True
                current_label = self.majority_label

            # Score rules
            if current_label == self.minority_label and is_clean_generated:
                is_clean_generated = False
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
                is_buggy_generated = False
            elif current_label == self.majority_label:
                score -= 1 * (i + 1)

        return credible_cf_instance, score

    def fit(self, X, y):
        # Store original data for FAISS searches
        self.original_X = X.copy()
        self.original_y = y.copy()

        # Train FAISS on original data
        self.model = self._train_faiss_model(self.original_X.astype('float32'))

        # Step 1: Majority-to-minority switching (M2MS) - Now with deletion
        X_new, y_new = self._m2ms(X, y)
        clean_data, buggy_data = self._data_separating(X_new, y_new)

        # If after deletion, minority is already majority or balanced, return early
        if buggy_data.shape[0] >= clean_data.shape[0]:
            return X_new, y_new, [], 0
        else:
            num_need = clean_data.shape[0] - buggy_data.shape[0]

        # Step 2: Majority-with-minority oversampling (MMOS)
        cf_item = Parallel(n_jobs=-1)(
            delayed(self._process_cf_item)(buggy_data[i], clean_data, y_new) for i in range(buggy_data.shape[0]))

        return X_new, y_new, cf_item, num_need

    def _cfss(self, X, y, cf_item, num_need):
        cf_item = {i: item for i, item in enumerate(cf_item) if item}
        processed_cf_item = {k: sorted(v, key=lambda x: x[2], reverse=True) for k, v in cf_item.items() if v}

        if not processed_cf_item:
            return X, y, processed_cf_item

        total_count = sum(len(items) for items in processed_cf_item.values())
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
        """
        X_new, y_new, cf_item, num_need = self.fit(X, y)

        if num_need == 0:
            return X_new, y_new

        X_balance, y_balance, _ = self._cfss(X_new, y_new, cf_item, num_need)

        return X_balance, y_balance
