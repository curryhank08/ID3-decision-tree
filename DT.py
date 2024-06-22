''' 
author: H24111057 姚博瀚
'''

import numpy as np
import pandas as pd

# Decision Tree Node class
class Node:
    def __init__(self, entropy, num_samples, num_samples_per_class, predicted_class):
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.children = {}

# Decision Tree class
class DecisionTree:
    def __init__(self, max_depth = 30):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _entropy(self, y):
        m = len(y)
        class_probs = [np.sum(y == c) / m for c in range(self.n_classes_)]
        return -np.sum(p * np.log2(p) if p > 0 else 0 for p in class_probs)

    def _information_gain(self, y, y_left, y_right):
        ent_parent = self._entropy(y)
        ent_left = self._entropy(y_left)
        ent_right = self._entropy(y_right)
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        return ent_parent - (weight_left * ent_left + weight_right * ent_right)
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        #ent_parent = self._entropy(y)
        best_info_gain = 0
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            feature_values = X.iloc[:, idx]
            unique_values = feature_values.unique()

            # If the feature is binary, consider only one threshold
            if len(unique_values) == 2:
                thresholds = [unique_values[0]]
            else:
                # For continuous or discrete features, consider midpoints between unique values
                thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

            for thr in thresholds:
                # Split the dataset based on the current threshold
                y_left = y[feature_values < thr]
                y_right = y[feature_values >= thr]

                # Calculate information gain
                info_gain = self._information_gain(y, y_left, y_right)

                # Update best split if current information gain is higher
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            entropy=self._entropy(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                feature_values = X.iloc[:, idx]
                X_left, y_left = X[feature_values < thr], y[feature_values < thr]
                X_right, y_right = X[feature_values >= thr], y[feature_values >= thr]
                node.feature_index = idx
                node.threshold = thr
                node.children['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node.children['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X.values]

    def _predict(self, inputs):
        node = self.tree_
        while 'left' in node.children:
            feature_value = inputs[node.feature_index]

            # Check if the feature value is numeric
            if isinstance(feature_value, (int, float)):
                if feature_value < node.threshold:
                    node = node.children['left']
                else:
                    node = node.children['right']
            else:
                # Handle non-numeric feature values
                raise ValueError("Unsupported feature type in decision tree prediction.")

        return node.predicted_class
    
    
    def print_tree_structure(self):
        tree_structure = self._to_string()
        print(tree_structure)

    def _to_string(self, node=None, depth=0):
        if node is None:
            node = self.tree_

        tree_str = ""
        indent = "  " * depth
        tree_str += f"{indent}Entropy: {node.entropy}\n"
        tree_str += f"{indent}Samples: {node.num_samples}\n"
        tree_str += f"{indent}Class Distribution: {node.num_samples_per_class}\n"
        tree_str += f"{indent}Predicted Class: {node.predicted_class}\n"

        if 'left' in node.children:
            tree_str += f"{indent}Decision: If feature {node.feature_index} < {node.threshold}\n"
            tree_str += self._to_string(node.children['left'], depth + 1)

            tree_str += f"{indent}Decision: If feature {node.feature_index} >= {node.threshold}\n"
            tree_str += self._to_string(node.children['right'], depth + 1)

        return tree_str
    

import random
# Pre-Pruned Decision Tree class
class PrePrunedDecisionTree(DecisionTree):
    def __init__(self, max_depth=30, min_samples_split=4, min_samples_leaf=2, min_info_gain=0.05, max_features=None):
        super().__init__(max_depth=max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_info_gain = min_info_gain
        self.max_features = max_features

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_info_gain = 0
        best_idx, best_thr = None, None

        # Select a random subset of features if max_features is specified
        if self.max_features is not None:
            all_features = range(self.n_features_)
            selected_features = random.sample(all_features, min(self.max_features, self.n_features_))
        else:
            selected_features = range(self.n_features_)

        for idx in selected_features:
            feature_values = X.iloc[:, idx]
            unique_values = feature_values.unique()

            # If the feature is binary, consider only one threshold
            if len(unique_values) == 2:
                thresholds = [unique_values[1]]
            else:
                # For continuous or discrete features, consider midpoints between unique values
                thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

            for thr in thresholds:
                # Split the dataset based on the current threshold
                y_left = y[feature_values < thr]
                y_right = y[feature_values >= thr]
                
                # Apply pre-pruning conditions
                if (len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf):
                    continue

                # Calculate information gain
                info_gain = self._information_gain(y, y_left, y_right)

                # Update best split if current information gain is higher
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_idx = idx
                    best_thr = thr
                    
        # Apply pre-pruning conditions
        if best_info_gain < self.min_info_gain:
            return None, None

        return best_idx, best_thr
    
        
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            entropy=self._entropy(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                feature_values = X.iloc[:, idx]
                X_left, y_left = X[feature_values < thr], y[feature_values < thr]
                X_right, y_right = X[feature_values >= thr], y[feature_values >= thr]
                
                # Apply pre-pruning conditions
                if (
                    len(y_left) >= self.min_samples_split
                    and len(y_right) >= self.min_samples_split
                ):
                    node.feature_index = idx
                    node.threshold = thr
                    node.children['left'] = self._grow_tree(X_left, y_left, depth + 1)
                    node.children['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node
