"""
Decision Tree Regressor - Step by Step Learning
Comprehensive Implementation for Interview Preparation

=========================================
#list all functions and classes in the file

1. compute_mse
2. compute_mae
3. _compute_impurity
4. _information_gain
5. _find_best_split
6. _create_leaf
7. _build_tree
8. fit
9. predict
10. fit_predict
11. score
12. get_depth
13. get_n_leaves
14. print_tree
15. get_params
16. set_params
"""
import numpy as np


class DecisionTreeRegressor:
    """
    Decision Tree Regressor with comprehensive functionality
    Following scikit-learn style interface for interview preparation
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='mse', random_state=None, verbose=False):
        """
        Initialize Decision Tree Regressor
        
        Parameters:
        -----------
        max_depth : int, default=None
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node
        criterion : str, default='mse'
            Function to measure the quality of a split ('mse', 'mae')
        random_state : int, default=None
            Random seed for reproducibility
        verbose : bool, default=False
            Whether to print training information
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.verbose = verbose
        
        # Model state attributes
        self.tree_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.is_fitted_ = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @staticmethod
    def compute_mse(y: np.ndarray) -> float:
        """Compute Mean Squared Error (variance) of target values"""
        if y.size == 0:
            return 0.0
        
        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)
        return float(mse)
    
    @staticmethod
    def compute_mae(y: np.ndarray) -> float:
        """Compute Mean Absolute Error of target values"""
        if y.size == 0:
            return 0.0
        
        median_y = np.median(y)
        mae = np.mean(np.abs(y - median_y))
        return float(mae)
    
    def _compute_impurity(self, y: np.ndarray) -> float:
        """Compute impurity using specified criterion"""
        if self.criterion == 'mse':
            return self.compute_mse(y)
        elif self.criterion == 'mae':
            return self.compute_mae(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, parent_y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """Calculate information gain from splitting parent into left/right"""
        n = parent_y.size
        if n == 0:
            return 0.0
        
        parent_impurity = self._compute_impurity(parent_y)
        
        # Calculate weighted average impurity of children
        left_weight = left_y.size / n
        right_weight = right_y.size / n
        
        left_impurity = self._compute_impurity(left_y)
        right_impurity = self._compute_impurity(right_y)
        
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
    
    def _create_leaf(self, y: np.ndarray):
        """Create a leaf - just return mean/median value"""
        # Use mean for MSE, median for MAE
        if self.criterion == 'mse':
            leaf_value = np.mean(y)
        else:  # mae
            leaf_value = np.median(y)
        return {"type": "leaf", "value": float(leaf_value)}
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Build tree recursively - simplified for interview"""
        # Stopping conditions
        if len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return self._create_leaf(y)
        
        if len(np.unique(y)) == 1:  # All same value
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
            print(f"Training Decision Tree Regressor with {len(X)} samples, {X.shape[1]} features...")
            print(f"Criterion: {self.criterion}, Max depth: {self.max_depth}")
        
        # Store training information
        self.feature_names_ = feature_names
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
            prediction = node["value"]
            predictions.append(prediction)
        
        if self.verbose:
            print("Predictions completed!")
        
        return np.array(predictions)
    
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
            Predicted target values
        """
        return self.fit(X, y, feature_names).predict(X)
    
    def score(self, X, y):
        """
        Return the R² score on the given test data and labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True target values
        
        Returns:
        --------
        score : float
            R² score (coefficient of determination)
        """
        predictions = self.predict(X)
        y_mean = np.mean(y)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2_score = 1 - (ss_res / ss_tot)
        return float(r2_score)
    
    def get_depth(self):
        """Get the depth of the tree"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting depth.")
        
        def _get_depth(node):
            if node["type"] == "leaf":
                return 0
            return 1 + max(_get_depth(node["left"]), _get_depth(node["right"]))
        
        return _get_depth(self.tree_)
    
    def get_n_leaves(self):
        """Get the number of leaves in the tree"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting number of leaves.")
        
        def _count_leaves(node):
            if node["type"] == "leaf":
                return 1
            return _count_leaves(node["left"]) + _count_leaves(node["right"])
        
        return _count_leaves(self.tree_)
    
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
            print(f"{indent}Leaf: value={node['value']:.3f}")
        else:
            feature_name = f"feature_{node['feature_index']}"
            if feature_names is not None and node['feature_index'] < len(feature_names):
                feature_name = feature_names[node['feature_index']]
            
            print(f"{indent}{feature_name} <= {node['threshold']:.2f}")
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
# DEMONSTRATION: Basic Decision Tree Regressor Usage
# ============================================================================

if __name__ == "__main__":
    # Sample regression dataset
    X_sample = np.array([
        [22, 65], [28, 80], [21, 72], [35, 85], [24, 60],
        [18, 55], [30, 90], [27, 70], [20, 50], [26, 65]
    ], dtype=float)

    # Create continuous target values (house prices based on temperature and humidity)
    y_sample = np.array([250000, 320000, 230000, 380000, 280000,
                         200000, 350000, 310000, 180000, 290000], dtype=float)

    feature_names = ["temperature", "humidity"]

    print("\n" + "="*60)
    print("DEMONSTRATION: Decision Tree Regressor")
    print("="*60)

    # Create and train the model
    print("\n1. Creating and training Decision Tree Regressor:")
    dt_regressor = DecisionTreeRegressor(max_depth=3, criterion='mse', verbose=True)

    # Train the model
    dt_regressor.fit(X_sample, y_sample, feature_names=feature_names)

    print(f"\n2. Model training completed!")
    print(f"   - Tree depth: {dt_regressor.get_depth()}")
    print(f"   - Number of leaves: {dt_regressor.get_n_leaves()}")

    # Make predictions
    print(f"\n3. Making predictions:")
    predictions = dt_regressor.predict(X_sample)
    print(f"   - Predictions: {predictions}")
    print(f"   - True values: {y_sample}")
    print(f"   - R² Score: {dt_regressor.score(X_sample, y_sample):.3f}")

    # Show tree structure
    print(f"\n4. Tree structure:")
    dt_regressor.print_tree(feature_names=feature_names)

    # Test different parameters
    print(f"\n5. Testing different criteria:")
    for criterion in ['mse', 'mae']:
        dt_test = DecisionTreeRegressor(max_depth=2, criterion=criterion, verbose=False)
        dt_test.fit(X_sample, y_sample)
        r2 = dt_test.score(X_sample, y_sample)
        print(f"   - {criterion.upper()} criterion: R²={r2:.3f}")

    # Test different max depths
    print(f"\n6. Testing different max depths:")
    for max_depth in [1, 2, 3, None]:
        dt_test = DecisionTreeRegressor(max_depth=max_depth, verbose=False)
        dt_test.fit(X_sample, y_sample)
        r2 = dt_test.score(X_sample, y_sample)
        depth = dt_test.get_depth()
        leaves = dt_test.get_n_leaves()
        print(f"   - Max depth {max_depth}: R²={r2:.3f}, actual_depth={depth}, leaves={leaves}")

    # ============================================================================
    # INTERVIEW QUESTIONS & ANSWERS
    # ============================================================================

    print("\n" + "="*60)
    print("INTERVIEW QUESTIONS & ANSWERS")
    print("="*60)

    print("\n1. BASIC CONCEPT QUESTIONS:")
    print("Q: What is a Decision Tree Regressor?")
    print("A: A Decision Tree Regressor is a supervised learning algorithm that creates a tree-like")
    print("   model to predict continuous target values based on input features.")

    print("\nQ: How does Decision Tree Regressor differ from Classifier?")
    print("A: Regressor predicts continuous values, uses MSE/MAE for splitting, and creates")
    print("   leaves with mean/median values instead of class labels.")

    print("\nQ: What are the main parameters of Decision Tree Regressor?")
    print("A: max_depth, min_samples_split, min_samples_leaf, criterion, random_state")

    print("\n2. SPLITTING QUESTIONS:")
    print("Q: What is variance reduction?")
    print("A: Variance reduction measures how much a split reduces the variance/uncertainty")
    print("   in the target values. Higher reduction means better split.")

    print("\nQ: What's the difference between MSE and MAE criteria?")
    print("A: MSE uses mean squared error (sensitive to outliers), MAE uses mean absolute")
    print("   error (robust to outliers)")

    print("\nQ: How do you choose the best split for regression?")
    print("A: Try all possible splits and choose the one with maximum variance reduction")

    print("\n3. LEAF CREATION QUESTIONS:")
    print("Q: What value do you assign to leaf nodes in regression?")
    print("A: Mean of target values for MSE criterion, median for MAE criterion")

    print("\nQ: Why use mean vs median in leaves?")
    print("A: Mean minimizes MSE, median minimizes MAE. Choice depends on the criterion used.")

    print("\nQ: How do you handle outliers in Decision Tree Regression?")
    print("A: Use MAE criterion instead of MSE, or preprocess data to handle outliers")

    print("\n4. IMPLEMENTATION QUESTIONS:")
    print("Q: How would you implement Decision Tree Regressor from scratch?")
    print("A: 1) Define node structure 2) Implement variance functions 3) Find best splits")
    print("   4) Build tree recursively 5) Implement prediction traversal")

    print("\nQ: What's the time complexity of building a regression tree?")
    print("A: O(n * m * log(n)) where n=samples, m=features")

    print("\nQ: How would you handle missing values in regression trees?")
    print("A: Use surrogate splits, impute missing values, or handle in splitting logic")

    print("\n5. ADVANTAGES & DISADVANTAGES:")
    print("Q: What are the advantages of Decision Tree Regressors?")
    print("A: Easy to interpret, handle non-linear relationships, no scaling needed")

    print("\nQ: What are the disadvantages?")
    print("A: Prone to overfitting, unstable, can create step-like predictions")

    print("\nQ: When would you use Decision Tree Regressors?")
    print("A: When interpretability is important, for non-linear data, as baseline model")

    print("\n6. OPTIMIZATION QUESTIONS:")
    print("Q: How can you make Decision Tree Regressors more robust?")
    print("A: Use ensemble methods (Random Forest, Gradient Boosting)")

    print("\nQ: How would you handle overfitting in regression trees?")
    print("A: Use max_depth, min_samples_split, min_samples_leaf, or pruning")

    print("\nQ: How would you handle imbalanced regression targets?")
    print("A: Use weighted splitting, transform targets, or use robust criteria")

    print("\n7. REAL-WORLD QUESTIONS:")
    print("Q: How would you use Decision Tree Regressors for feature selection?")
    print("A: Use feature importance scores from trained trees")

    print("\nQ: How would you visualize a Decision Tree Regressor?")
    print("A: Use graphviz, matplotlib, or tree plotting libraries")

    print("\nQ: How would you deploy a Decision Tree Regressor model?")
    print("A: Export tree structure to JSON/dict, implement prediction logic in production")

    print("\n8. MATHEMATICAL QUESTIONS:")
    print("Q: What is the mathematical formula for variance reduction?")
    print("A: VR = Var(parent) - [n_left/n * Var(left) + n_right/n * Var(right)]")

    print("\nQ: What is MSE formula?")
    print("A: MSE = (1/n) * Σ(y_i - ŷ_i)²")

    print("\nQ: What is MAE formula?")
    print("A: MAE = (1/n) * Σ|y_i - ŷ_i|")

    print("\n9. COMPARISON QUESTIONS:")
    print("Q: How do Decision Tree Regressors compare to Linear Regression?")
    print("A: Trees handle non-linear relationships better, but are less interpretable")

    print("\nQ: How do Decision Tree Regressors compare to Neural Networks?")
    print("A: Trees are more interpretable but less flexible than neural networks")

    print("\nQ: When would you choose Decision Trees over other regression methods?")
    print("A: When interpretability is crucial and data has clear decision boundaries")

    print("\n10. DEBUGGING QUESTIONS:")
    print("Q: What if Decision Tree Regressor gives poor results?")
    print("A: Check data preprocessing, adjust parameters, try different criteria")

    print("\nQ: How would you debug Decision Tree Regressor implementation?")
    print("A: Compare with sklearn, check variance calculations, verify splitting logic")

    print("\nQ: What if predictions are too step-like?")
    print("A: Increase max_depth, reduce min_samples_leaf, or use ensemble methods")

    print("\n✅ Decision Tree Regressor Interview Q&A section completed!")
