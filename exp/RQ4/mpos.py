import numpy as np
from scipy.spatial.distance import cdist


class MPOS:

    def __init__(self):
        pass

    @staticmethod
    def fit_resample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        unique_labels, counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(counts)]
        majority_label = unique_labels[np.argmax(counts)]
        minority_samples = X[y == minority_label]
        majority_samples = X[y == majority_label]

        minority_count = minority_samples.shape[0]
        majority_count = majority_samples.shape[0]

        num_to_generate = majority_count - minority_count
        if num_to_generate <= 0:
            return X, y

        num_features = minority_samples.shape[1]
        synthetic_samples = np.zeros((num_to_generate, num_features))

        for i in range(num_to_generate):
            new_sample = np.zeros(num_features)
            for j in range(num_features):
                random_idx = np.random.randint(0, minority_count)
                current_sample = minority_samples[random_idx]
                distances = cdist([current_sample], minority_samples)
                sorted_indices = np.argsort(distances.flatten())

                r = np.random.rand()
                new_sample[j] = current_sample[j] + r * (current_sample[j] - minority_samples[sorted_indices[1], j])

            synthetic_samples[i] = new_sample

        balanced_X = np.vstack((X, synthetic_samples))
        balanced_y = np.hstack((y, np.ones(num_to_generate)))

        return balanced_X, balanced_y
