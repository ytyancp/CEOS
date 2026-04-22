from sklearn.metrics import confusion_matrix
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


def metric_sdp(true_set, preset):

    cm = confusion_matrix(true_set, preset)
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]

    pd = tp / (tp + fn)
    pf = fp / (tn + fp)
    balance = 1 - math.sqrt((0 - pf) * (0 - pf) + (1 - pd) * (1 - pd)) / math.sqrt(2)

    return pf, balance


def validity_score(cf_labels, desired_labels):

    if len(cf_labels) == 0:
        return 0.0

    valid_mask = cf_labels == desired_labels
    validity = np.mean(valid_mask)
    return validity


def proximity_score(original_samples, cf_samples, norm='l2'):

    if len(cf_samples) == 0:
        return 0.0

    if norm == 'l1':
        distances = pairwise_distances(original_samples, cf_samples, metric='manhattan')
    elif norm == 'l2':
        distances = pairwise_distances(original_samples, cf_samples, metric='euclidean')
    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    dialog_distances = np.diag(distances)
    proximity = np.mean(dialog_distances)
    return proximity


def plausibility_score(cf_samples, training_data, k=5):

    if len(cf_samples) == 0:
        return 0.0

    nns = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(training_data)
    distances, _ = nns.kneighbors(cf_samples)

    avg_distances = np.mean(distances, axis=1)
    plausibility = np.mean(avg_distances)
    return plausibility


def metric_cf(original_samples, cf_samples, original_labels, cf_labels, training_data, desired_labels):
    metrics = {'validity': validity_score(cf_labels, desired_labels),
               'proximity_l1': proximity_score(original_samples, cf_samples, norm='l1'),
               'proximity_l2': proximity_score(original_samples, cf_samples, norm='l2'),
               'plausibility': plausibility_score(cf_samples, training_data)}

    return metrics


def r2_score(global_model, local_model, X_neighbors):
    """
    Calculate R² score between global model and local model on a set of neighbors.

    Args:
        global_model: The complex black-box model
        local_model: The simple interpretable model
        X_neighbors: Neighborhood samples around the test instance

    Returns:
        R² score (higher is better, max = 1)
    """
    try:
        if hasattr(global_model, 'predict_proba'):
            local_probs = global_model.predict_proba(X_neighbors)
        else:
            local_probs = global_model.predict(X_neighbors)

        lr_probs = local_model.predict_proba(X_neighbors)

        ss_res = np.sum((local_probs - lr_probs) ** 2)
        ss_tot = np.sum((local_probs - np.mean(local_probs)) ** 2)

        return 1 - (ss_res / (ss_tot + 1e-9))

    except Exception as e:
        print(f"Error calculating R² score: {e}")
        return 0


def get_top_k_features_from_lr(lr_model, k=5):

    try:
        coefs = lr_model.coef_[0]
        top_indices = np.argsort(np.abs(coefs))[::-1][:k]
        return set(top_indices)
    except Exception as e:
        print(f"Error extracting top-k features from LR: {e}")
        return set()


def get_top_k_features_from_lime(lime_exp, predicted_label, k=5):

    try:
        local_map = lime_exp.as_map()[predicted_label]
        local_map_sorted = sorted(local_map, key=lambda x: abs(x[1]), reverse=True)[:k]
        feature_indices = [fm[0] for fm in local_map_sorted]
        return set(feature_indices)
    except Exception as e:
        print(f"Error extracting top-k features from LIME: {e}")
        return set()


def calc_jaccard_index(lime_exp, predicted_label, lr_model, k=5):

    lime_features = get_top_k_features_from_lime(lime_exp, predicted_label, k)
    lr_features = get_top_k_features_from_lr(lr_model, k)

    # Calculate Jaccard index
    inter = lime_features.intersection(lr_features)
    union = lime_features.union(lr_features)
    return len(inter) / len(union) if len(union) > 0 else 0.0

