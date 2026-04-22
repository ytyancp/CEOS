import re
import faiss
import random
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class CEOS:

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
        self.cfe_count = []

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
        Switch the potential majority samples to minority class.

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
                # Update if current neighbor is in 'overlapping' and has not yet switched to prevent duplicates
                if tuple(X[index]) in overlapping_samples and index not in switched_indices:
                    y_new[index] = self.minority_label
                    switched_indices.add(index)

        return X, y_new

    def _find_clean_borderline(self, clean_data, y):
        """
        Find the boundary sample in 'clean' class.

        param clean_data: Data whose label is 'majority_label'.
        param y: Class labels.
        :return: List of boundary samples.
        """
        boundary_samples = []

        for clean_sample in clean_data:
            distances, indices = self.model.search(clean_sample.reshape(1, -1), self.k_neighbors + 1)
            labels = y[indices[0][1:]]
            if np.sum(labels == self.majority_label) in [2, 3]:
                boundary_samples.append(clean_sample)

        return boundary_samples

    def _process_cf_item(self, buggy_instance, clean_data, y):
        """
        Process counterfactual instances to obtain credible counterfactual instances.

        param buggy_instance: An instance from the buggy class.
        param clean_data: Data whose label is 'majority_label'.
        :return: List items. Such as [[(clean_index1, cf_sample1, score1)], [(clean_index2, cf_sample2, score2)],...]].
        """
        items = []
        for j in range(len(clean_data)):
            cf_sample, score = self._mmos(clean_data[j], buggy_instance, y, desired_label=self.minority_label)
            if cf_sample is not None and score > 0:
                items.append((j, cf_sample, score))
        return items

    def _mmos(self, clean_instance, buggy_instance, y, desired_label):
        """
        Generate a credible counterfactual instance via a clean instance and a buggy instance.

        param clean_instance: An instance selected from clean class.
        param buggy_instance: An instance selected from buggy class.
        param y: Class labels.
        param desired_label: Desired label for the counterfactual instance.
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

            if desired_label == self.minority_label:
                # Score rules: 'clean': -1, 'buggy': 1, 'clean->buggy' or 'buggy->clean': 1
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
            else:
                # Score rules: 'clean': 1, 'buggy': -1, 'clean->buggy' or 'buggy->clean': 1
                if current_label == self.majority_label and is_buggy_generated:
                    is_buggy_generated = False  # for real-time tracking of class status
                    score += 2 * (i + 1)
                    if credible_cf_instance is None:
                        credible_cf_instance = current_instance
                    else:
                        credible_cf_instance = (credible_cf_instance + current_instance) / 2
                elif current_label == self.majority_label:
                    score += 1 * (i + 1)
                    if credible_cf_instance is not None:
                        credible_cf_instance = (credible_cf_instance + current_instance) / 2

                if current_label == self.minority_label and is_clean_generated:
                    is_clean_generated = False  # 'score -= 0 * (i + 1)' is omitted here.
                elif current_label == self.minority_label:
                    score -= 1 * (i + 1)

        return credible_cf_instance, score

    def fit(self, X, y):
        self.model = self._train_faiss_model(X)
        # Step 1: Majority-to-minority switching (M2MS)
        X_new, y_new = self._m2ms(X, y)
        clean_data, buggy_data = self._data_separating(X_new, y_new)

        if buggy_data.shape[0] >= clean_data.shape[0]:
            return X_new, y_new, None, 0
        else:
            num_need = clean_data.shape[0] - buggy_data.shape[0]
        # Step 2: Majority-with-minority oversampling (MMOS)
        clean_border_samples = self._find_clean_borderline(clean_data, y_new)
        cf_item = Parallel(n_jobs=-1)(
            delayed(self._process_cf_item)(buggy_data[i], clean_border_samples, y_new) for i in range(len(buggy_data)))

        return X_new, y_new, cf_item, num_need

    def _cfss(self, X, y, cf_item, num_need):
        """
        Counterfactual sample selection for credible and high-score instances.

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

    @staticmethod
    def draw(select_clean, select_buggy, cf_sample, top_features_indices, feature_names, blackbox):
        """
        A PoC(Proof-of-Concept) visualization to show the counterfactual explanations.

        param select_clean: An instance selected from clean class.
        param select_buggy: An instance selected from buggy class.
        param cf_sample: A credible counterfactual instance.
        param top_features_indices: List of top-k feature indices.
        param feature_names: Original data feature names list.
        param blackbox: Black-box classifier, such as a scikit-learn model or a TensorFlow model.
        """
        value_relations = []
        for _, feature_index in enumerate(top_features_indices):
            buggy_value = select_buggy[feature_index]
            clean_value = select_clean[feature_index]
            value_relations.append(clean_value > buggy_value)

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        def plot_explanation(ax, query_instance):
            x_lower, x_upper = -12, 12
            y_lower, y_upper = 0, 4.5
            ax.set_xlim(x_lower, x_upper)
            ax.set_ylim(y_lower, y_upper)

            risk_score = blackbox.predict_proba(query_instance.reshape(1, -1))[0, 1]
            prediction_color = 'green' if risk_score < 0.5 else 'red'
            risk_high = y_upper - 0.54
            ax.text(-0.48, risk_high, "Risk Score:", va='center', ha='right', fontsize=12)
            ax.barh(risk_high, risk_score * 10, height=0.25, color=prediction_color, alpha=0.6)
            ax.barh(risk_high, 10 - risk_score * 10, height=0.25, color='#D3D3D3', left=risk_score * 10)
            ax.text(risk_score * 10 + 0.5, risk_high, f"{risk_score:.0%}",
                    va='center', ha='left', fontsize=10, color=prediction_color)

            # Setting border
            border_start = (x_lower + 1, y_lower + 0.05)
            border_width = x_upper - x_lower - 2
            border_height = y_upper - 0.9
            ax.add_patch(plt.Rectangle(border_start, border_width, border_height,
                                       fill=False, edgecolor='black', linewidth=2))

            # Setting query
            prediction = 'buggy' if risk_score >= 0.5 else 'clean'
            prediction_height = border_height - 0.23
            query_text = ax.text(-1, prediction_height, "Why this instance is predicted as ",
                                 ha='center', fontsize=12, weight='bold')
            text_width = query_text.get_window_extent(renderer=fig.canvas.get_renderer()).width
            prediction_x = text_width / fig.dpi + 2.7
            ax.text(prediction_x, prediction_height, f'{prediction}?',
                    ha='left', fontsize=12, fontstyle='italic', weight='bold', color=prediction_color)
            current_height = prediction_height - 0.4

            # What-if analysis
            for i, index in enumerate(top_features_indices):
                factual_value = query_instance[index]
                cf_value = cf_sample[index]
                feature_name = feature_names[index]
                relation = "equal"
                change = None
                difference = 0

                if factual_value > cf_value:
                    relation = "larger"
                    change = "Decrease"
                    difference = factual_value - cf_value
                elif factual_value < cf_value:
                    relation = "smaller"
                    change = "Increase"
                    difference = cf_value - factual_value

                max_value = max(factual_value, cf_value)
                if max_value < 1:
                    max_value = 1
                else:
                    max_value = (int(max_value * 1.2) // 10 + 1) * 10
                norm_actual = 10 * factual_value / max_value
                norm_cf = 10 * cf_value / max_value

                ax.text(x_lower + 1.25, current_height,
                        f"#{i + 1} The value of ", fontsize=10, ha='left')
                words = re.findall(r'[A-Z]+|[a-z]+', feature_name)
                abbreviation = ''.join(word[0].upper() for word in words)
                if not any(char.isupper() for char in feature_name) or len(feature_name) <= 3:
                    abbr_name = feature_name
                else:
                    abbr_name = abbreviation[:3]
                print(f'{feature_name} abbreviation: {abbr_name}')
                x_text = ax.text(-5.8, current_height, f"{abbr_name}",
                                 fontstyle='italic', fontsize=10, weight='bold')
                x_width = x_text.get_window_extent(renderer=fig.canvas.get_renderer()).width

                if relation == "equal":
                    ax.text(-4.7 + x_width / fig.dpi, current_height, f"is equivalent to counterfactual.",
                            fontsize=10, ha='left')
                else:
                    if risk_score < 0.5:
                        ax.text(-4.5 + x_width / fig.dpi, current_height,
                                f"is {relation}. {change} maximum △x by {difference:.2f}.", fontsize=10, ha='left')
                    else:
                        ax.text(-4.5 + x_width / fig.dpi, current_height,
                                f"is {relation}. {change} minimum △x by {difference:.2f}.", fontsize=10, ha='left')

                current_height -= 0.3
                bar_color = ('red', 'green') if value_relations[i] else ('green', 'red')
                ax.text(-0.7, current_height, f"Factual = {factual_value:.2f}", fontsize=9, ha='right')
                current_height -= 0.2
                ax.barh(current_height + 0.25, norm_cf, height=0.15, color=bar_color[0], alpha=0.6)
                ax.barh(current_height + 0.25, 10 - norm_cf, height=0.15, color=bar_color[1], left=norm_cf)
                current_height -= 0.15
                ax.text(-0.7, current_height, f"Counterfactual = {cf_value:.2f}", fontsize=9, ha='right')

                for k in range(11):
                    scale = k * max_value / 10
                    if max_value == 1:
                        ax.text(k, current_height, f"{scale:.1f}", ha='center', fontsize=7.2, fontstyle='italic')
                    else:
                        ax.text(k, current_height, f"{scale:.0f}", ha='center', fontsize=7.2, fontstyle='italic')
                ax.plot([norm_actual, norm_actual], [current_height + 0.3, current_height + 0.55],
                        color='black', linewidth=2.5)
                current_height -= 0.4

            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')

        plot_explanation(axes[0], select_clean)
        plot_explanation(axes[1], select_buggy)

        plt.tight_layout()
        plt.show()

    def fit_resample(self, X, y):
        """
        Fit and resample the training data.

        :return: Resampled dataset.
        """
        X_new, y_new, cf_item, num_need = self.fit(X, y)
        if num_need == 0:
            return X_new, y_new
        # step 3: Counterfactual sample selection
        X_balance, y_balance, _ = self._cfss(X_new, y_new, cf_item, num_need)

        return X_balance, y_balance

    def _process_cfe(self, query_instance, clean_data, buggy_data, blackbox, y):
        """
        Process counterfactual explanations.

        param query_instance: An instance to explain.
        param clean_data: Data whose label is 'majority_label'.
        param buggy_data: Data whose label is 'minority_label'.
        param blackbox: Black-box classifier, such as a scikit-learn model or a TensorFlow model.
        param y: Class labels.
        """
        proba = blackbox.predict_proba(query_instance.reshape(1, -1))[0, 1]

        if proba > 0.5:
            cfs = []
            for j in range(len(clean_data)):
                cf_sample, score = self._mmos(clean_data[j], query_instance, y, desired_label=self.majority_label)
                if cf_sample is not None and score > 0:
                    cfs.append((j, cf_sample, score, self.majority_label))

        else:
            cfs = []
            for i in range(len(buggy_data)):
                cf_sample, score = self._mmos(query_instance, buggy_data[i], y, desired_label=self.minority_label)
                if cf_sample is not None and score > 0:
                    cfs.append((i, cf_sample, score, self.minority_label))
        return cfs

    def fit_explanation(self, X_train, y_train, X_test, feature_names, global_model=None, top_k=5, is_plot=False):
        """
        Fit and visualize explanation.

        param X_train: Training dataset.
        param y_train: Class labels.
        param X_test: Testing dataset.
        param feature_names: List of feature names.
        param blackbox: Black-box classifier, such as an ML model or DL model.
        param top_k: Number of top features to select.
        param is_plot: Whether to plot the explanation or not.
        :return: Dict items like {'test_index1':[[(clean_index1/buggy_index1, cf_sample1, score1, desired_label1)]...}.
        """

        clean_data, buggy_data = self._data_separating(X_train, y_train)

        X_balance, y_balance = self.fit_resample(X_train, y_train)
        if global_model is None:
            global_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        global_model.fit(X_balance, y_balance)

        # Generate counterfactual for query instances (f(cf)!=f(query))
        cfe = Parallel(n_jobs=-1)(delayed(self._process_cfe)(X_test[i], clean_data, buggy_data, global_model, y_balance)
                                  for i in range(len(X_test)))
        cfe_items = {i: items for i, items in enumerate(cfe) if items}
        processed_cfe_items = {k: sorted(v, key=lambda x: x[2], reverse=True) for k, v in cfe_items.items() if v}

        # Count the number of counterfactuals for each query instance.
        for i in range(len(X_test)):
            count = len(processed_cfe_items.get(i, []))
            self.cfe_count.append(count)

        query_idx = random.randint(0, len(X_test) - 1)
        first_round_cfs = processed_cfe_items[query_idx]
        X_local = []  # dataset for local model (a simple, interpretable model, e.g., a linear or rule-based predictor)
        y_local = []  # labels for local model

        for cf_info in first_round_cfs:
            # Generate borderline factual for query instances (f(bf)=f(query)), we apply a reverse perturbation.
            second_round_bfs = self._process_cfe(cf_info[1], clean_data, buggy_data, global_model, y_balance)

            if second_round_bfs:
                sorted_bfs = sorted(second_round_bfs, key=lambda x: x[2], reverse=True)
                best_bf = sorted_bfs[0]
                X_local.append(best_bf[1])
                y_local.append(best_bf[3])
            X_local.append(cf_info[1])
            y_local.append(cf_info[3])

        X_local = np.array(X_local) if X_local else np.array([])
        y_local = np.array(y_local) if y_local else np.array([])
        lr_local = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000)
        lr_local.fit(X_local, y_local)

        coef = lr_local.coef_[0]
        feature_importance = np.abs(coef)
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_features = [(feature_names[i], coef[i]) for i in sorted_indices[:top_k]]
        top_indices = sorted_indices[:top_k]

        if is_plot:
            select_clean = select_buggy = X_test[query_idx]
            random_item = random.choice(processed_cfe_items[query_idx])
            if random_item[3] == self.majority_label:
                select_clean = clean_data[random_item[0]]
            else:
                select_buggy = buggy_data[random_item[0]]
            select_cfe = random_item[1]
            self.draw(select_clean, select_buggy, select_cfe, top_indices, feature_names, global_model)

        return X_balance, y_balance, processed_cfe_items, top_features
