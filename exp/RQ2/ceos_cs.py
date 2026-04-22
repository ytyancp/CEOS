#  Only the class switching stage is implemented in this file.
import numpy as np
import faiss


class CS:

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

    def _m2ms(self, X, y):
        """
        Separate the dataset into majority and minority classes.

        param X: Feature dataset.
        param y: Class labels.
        :return: Clean and buggy class data.
        """

        clean_data, buggy_data = self._data_separating(X, y)
        y_new = y.copy()

        switched_indices = set()
        overlapping_samples = set()

        for clean_sample in clean_data:
            distances, indices = self.model.search(clean_sample.reshape(1, -1), self.k_neighbors + 1)
            labels = y[indices[0][1:]]
            minority_count = np.sum(labels == self.minority_label)

            if minority_count > self.k_neighbors // 2:
                overlapping_samples.add(tuple(clean_sample))

        for buggy_sample in buggy_data:
            distances, indices = self.model.search(buggy_sample.reshape(1, -1), self.k_neighbors + 1)

            for index in indices[0][1:]:
                if index not in switched_indices and tuple(X[index]) in overlapping_samples:
                    y_new[index] = self.minority_label
                    switched_indices.add(index)

        return X, y_new

    def fit_resample(self, X, y):
        self.model = self._train_faiss_model(X)
        X_new, y_new = self._m2ms(X, y)
        return X_new, y_new
