"""
Decision Tree Classifier - Step by Step Learning
Comprehensive Implementation for Interview Preparation

=========================================
#list all functions and classes in the file and give one line description of each function and class
1. DecisionTree - Decision Tree Classifier with comprehensive functionality
Following scikit-learn style interface for interview preparation
1. compute_entropy - Compute Shannon entropy of labels = -sum(p_i * log2(p_i))
2. compute_gini -  1 - sum(p_i^2)
3. _compute_impurity - Compute impurity using specified criterion
4. _information_gain = parent impurity - weighted average of child impurity - if ig is highest, then split is best
5. _find_best_split - Find the best split for the given data
# write description of candidate thresholds in the code
candidate thresholds are the midpoints between sorted unique values of the feature
for example, if the feature values are [1, 2, 3, 4, 5], then the candidate thresholds are [1.5, 2.5, 3.5, 4.5]
if we use unique values, most values will fall on boundaries, so we use midpoints between sorted unique valuesFe
then for each candidate threshold, we split the data into left and right based on the threshold
and calculate the information gain
if the information gain is highest, then the split is best
then we select the candidate threshold with the highest information gain

6. _create_leaf - Create a leaf node with the most common class
7. _build_tree - Recursively build the decision tree
8. fit - Fit the Decision Tree to training data
9. predict - Predict class labels for test samples
10. predict_proba - Predict class probabilities for test samples
11. fit_predict - Fit the model and predict on the same data
12. score - Return the mean accuracy on the given test data and labels
13. get_depth - Get the depth of the tree
14. get_n_leaves - Get the number of leaves in the tree
15. print_tree - Print the tree structure
16. get_params - Get parameters for this estimator
17. set_params - Set parameters for this estimator
 
"""
import numpy as np
from collections import Counter


class DecisionTree:
    """
    Decision Tree Classifier with comprehensive functionality
    Following scikit-learn style interface for interview preparation
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', random_state=None, verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.verbose = verbose
        
        # Model state attributes
        self.tree_ = None
        self.feature_names_ = None
        self.classes_ = None
        self.n_features_ = None
        self.is_fitted_ = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @staticmethod
    def compute_entropy(labels: np.ndarray) -> float:
        """Compute Shannon entropy of labels"""
        if labels.size == 0:
            return 0.0
        
        # Get unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / labels.size
        
        # Calculate entropy
        entropy = 0.0
        for prob in probabilities:
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return float(entropy)
    
    @staticmethod
    def compute_gini(labels: np.ndarray) -> float:
        """Compute Gini impurity of labels"""
        if labels.size == 0:
            return 0.0
        
        # Get unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / labels.size
        
        # Calculate Gini impurity
        gini = 1.0
        for prob in probabilities:
            gini -= prob ** 2
        
        return float(gini)
    
    def _compute_impurity(self, labels: np.ndarray) -> float:
        """Compute impurity using specified criterion"""
        if self.criterion == 'gini':
            return self.compute_gini(labels)
        elif self.criterion == 'entropy':
            return self.compute_entropy(labels)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, parent_labels: np.ndarray, left_labels: np.ndarray, right_labels: np.ndarray) -> float:
        """Calculate information gain from splitting parent into left/right"""
        n = parent_labels.size
        if n == 0:
            return 0.0
        
        parent_impurity = self._compute_impurity(parent_labels)
        
        # Calculate weighted average impurity of children
        left_weight = left_labels.size / n
        right_weight = right_labels.size / n
        
        left_impurity = self._compute_impurity(left_labels)
        right_impurity = self._compute_impurity(right_labels)
        
        information_gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)
        return float(information_gain)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best split - simplified for interview"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        num_samples, num_features = X.shape
        
        # Try each feature
        for feature_idx in range(num_features):
            values = X[:, feature_idx]
            unique_vals = np.unique(values)
            
            # Try midpoints as thresholds
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            
            for threshold in thresholds:
                # Split data
                left_temp = values <= threshold
                left_y = y[left_temp]
                right_y = y[~left_temp]
                
                # Skip if split is too small
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, left_y, right_y)
                
                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _create_leaf(self, labels: np.ndarray):
        """Create a leaf - just return most common class"""
        class_counts = Counter(labels)
        most_common_class = class_counts.most_common(1)[0][0]
        return {"type": "leaf", "class": int(most_common_class)}
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Build tree recursively - simplified for interview"""
        # Stopping conditions
        if len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return self._create_leaf(y)
        
        if len(np.unique(y)) == 1:  # All same class
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # No good split found
        if best_feature is None:
            return self._create_leaf(y)
        
        # Split data
        left_temp = X[:, best_feature] <= best_threshold
        X_left, y_left = X[left_temp], y[left_temp]
        X_right, y_right = X[~left_temp], y[~left_temp]
        
        # Build subtrees
        return {
            "type": "internal",
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1)
        }
    
    def fit(self, X, y, feature_names=None):
        X = np.array(X)
        y = np.array(y)
        
        if self.verbose:
            print(f"Training Decision Tree with {len(X)} samples, {X.shape[1]} features...")
            print(f"Criterion: {self.criterion}, Max depth: {self.max_depth}")
        
        # Store training information
        self.feature_names_ = feature_names
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        
        # Build the tree
        self.tree_ = self._build_tree(X, y)
        self.is_fitted_ = True
        
        if self.verbose:
            print("Training completed! Tree is ready for predictions.")
        
        return self
    
    def predict(self, X):

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.array(X)
        predictions = []
        
        if self.verbose:
            print(f"Making predictions for {len(X)} test samples...")
        
        for x in X:
            # Navigate the tree for this sample
            node = self.tree_
            while node["type"] != "leaf":
                feature_idx = node["feature_index"]
                threshold = node["threshold"]
                
                if x[feature_idx] <= threshold:
                    node = node["left"]
                else:
                    node = node["right"]
            
            # At leaf node, get prediction
            prediction = node["class"]
            predictions.append(prediction)
        
        if self.verbose:
            print("Predictions completed!")
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for test samples
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        probabilities : array of shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.array(X)
        n_classes = len(self.classes_)
        probabilities = []
        
        if self.verbose:
            print(f"Computing probabilities for {len(X)} test samples...")
        
        for x in X:
            # Navigate to leaf and get class distribution
            node = self.tree_
            while node["type"] != "leaf":
                feature_idx = node["feature_index"]
                threshold = node["threshold"]
                
                if x[feature_idx] <= threshold:
                    node = node["left"]
                else:
                    node = node["right"]
            
            # Get class probabilities from leaf
            class_dist = node.get("class_distribution", {node["class"]: node["samples"]})
            total_samples = sum(class_dist.values())
            
            prob_vector = np.zeros(n_classes)
            for i, class_label in enumerate(self.classes_):
                prob_vector[i] = class_dist.get(class_label, 0) / total_samples
            
            probabilities.append(prob_vector)
        
        if self.verbose:
            print("Probability computation completed!")
        
        return np.array(probabilities)
    
    def fit_predict(self, X, y, feature_names=None):
        """
        Fit the model and predict on the same data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training samples
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        predictions : array of shape (n_samples,)
            Predicted class labels
        """
        return self.fit(X, y, feature_names).predict(X)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
        
        Returns:
        --------
        score : float
            Mean accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    

    
    def print_tree(self, node=None, depth=0, feature_names=None):
        """Print the tree structure"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before printing tree.")
        
        if node is None:
            node = self.tree_
        
        if feature_names is None:
            feature_names = self.feature_names_
        
        indent = "  " * depth
        
        if node["type"] == "leaf":
            class_dist = node.get("class_distribution", {node["class"]: node["samples"]})
            print(f"{indent}Leaf: class={node['class']}, samples={node['samples']}, dist={class_dist}")
        else:
            feature_name = f"feature_{node['feature_index']}"
            if feature_names is not None and node['feature_index'] < len(feature_names):
                feature_name = feature_names[node['feature_index']]
            
            print(f"{indent}{feature_name} <= {node['threshold']:.2f} [IG={node['information_gain']:.3f}, samples={node['samples']}]")
            self.print_tree(node["left"], depth + 1, feature_names)
            self.print_tree(node["right"], depth + 1, feature_names)
    
    def get_params(self):
        """Get parameters for this estimator"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self


# ============================================================================
# DEMONSTRATION: Basic Decision Tree Usage
# ============================================================================

if __name__ == "__main__":
    # Sample dataset
    X_sample = np.array([
        [22, 65], [28, 80], [21, 72], [35, 85], [24, 60],
        [18, 55], [30, 90], [27, 70], [20, 50], [26, 65]
    ], dtype=float)

    y_sample = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1], dtype=int)

    feature_names = ["temperature", "humidity"]

    print("\n" + "="*60)
    print("DEMONSTRATION: Decision Tree Classifier")
    print("="*60)

    # Create and train the model
    print("\n1. Creating and training Decision Tree model:")
    dt_model = DecisionTree(max_depth=3, criterion='gini', verbose=True)

    # Train the model
    dt_model.fit(X_sample, y_sample, feature_names=feature_names)

    print(f"\n2. Model training completed!")
    print(f"   - Tree depth: {dt_model.get_depth()}")
    print(f"   - Number of leaves: {dt_model.get_n_leaves()}")
    print(f"   - Classes: {dt_model.classes_}")

    # Make predictions
    print(f"\n3. Making predictions:")
    predictions = dt_model.predict(X_sample)
    print(f"   - Predictions: {predictions}")
    print(f"   - True labels: {y_sample}")
    print(f"   - Accuracy: {dt_model.score(X_sample, y_sample):.3f}")

    # Show tree structure
    print(f"\n4. Tree structure:")
    dt_model.print_tree(feature_names=feature_names)









