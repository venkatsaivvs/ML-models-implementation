"""
K-Nearest Neighbors Regressor - Step by Step Learning
Comprehensive Implementation for Interview Preparation

=========================================
#list all functions and classes in the file

1. euclidean_distance
2. manhattan_distance
3. minkowski_distance
4. euclidean_distance_vectorized
5. manhattan_distance_vectorized
6. _compute_distance
7. _compute_distances_vectorized
8. fit
9. predict
10. fit_predict
11. score
12. get_params
13. set_params
=========================================
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate sample data for demonstration
np.random.seed(42)

# Create regression dataset
X_reg, y_reg = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor with comprehensive functionality
    Following scikit-learn style interface for interview preparation
    """
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform', verbose=False):
        """
        Initialize KNN regressor
        
        Parameters:
        -----------
        k : int, default=3
            Number of nearest neighbors to consider
        distance_metric : str, default='euclidean'
            Distance metric to use ('euclidean', 'manhattan', 'minkowski')
        weights : str, default='uniform'
            Weight function used in prediction ('uniform', 'distance')
        verbose : bool, default=False
            Whether to print training information
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.verbose = verbose
        
        # Model state attributes
        self.X_train_ = None
        self.y_train_ = None
        self.is_fitted_ = False
        
        if self.verbose:
            print(f"Initialized KNN Regressor with k={k}, distance_metric='{distance_metric}', weights='{weights}'")
    
    @staticmethod
    def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute Euclidean distance between two points"""
        diff = point1 - point2
        return float(np.sqrt(np.sum(diff ** 2)))

    @staticmethod
    def manhattan_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute Manhattan distance between two points"""
        diff = point1 - point2
        return float(np.sum(np.abs(diff)))

    @staticmethod
    def minkowski_distance(point1: np.ndarray, point2: np.ndarray, p: int = 2) -> float:
        """Compute Minkowski distance between two points"""
        diff = point1 - point2
        return float(np.sum(np.abs(diff) ** p) ** (1/p))

    @staticmethod
    def euclidean_distance_vectorized(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance computation"""
        return np.linalg.norm(X1 - X2, axis=1)

    @staticmethod
    def manhattan_distance_vectorized(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Vectorized Manhattan distance computation"""
        return np.sum(np.abs(X1 - X2), axis=1)

    def _compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute distance between two points using specified metric"""
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(point1, point2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(point1, point2)
        elif self.distance_metric == 'minkowski':
            return self.minkowski_distance(point1, point2, p=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _compute_distances_vectorized(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Vectorized distance computation"""
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance_vectorized(X1, X2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance_vectorized(X1, X2)
        else:
            raise ValueError(f"Vectorized computation not implemented for: {self.distance_metric}")

    def fit(self, X, y):
        """
        Fit the KNN regressor to training data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training samples
        y : array-like of shape (n_samples,)
            Target values (continuous)
        
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.array(X)
        y = np.array(y)
        
        if self.verbose:
            print(f"Training KNN Regressor with {len(X)} samples, {X.shape[1]} features...")
            print(f"Target range: [{np.min(y):.2f}, {np.max(y):.2f}]")
        
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.is_fitted_ = True
        
        if self.verbose:
            print("Training completed! Model is ready for predictions.")
        
        return self

    def predict(self, X):
        """
        Predict target values for test samples
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        predictions : array of shape (n_samples,)
            Predicted target values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.array(X)
        predictions = []
        
        if self.verbose:
            print(f"Making predictions for {len(X)} test samples...")
        
        for test_point in X:
            # Compute distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train_):
                dist = self._compute_distance(test_point, train_point)
                distances.append((dist, self.y_train_[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_neighbors = distances[:self.k]
            
            # Extract target values and distances
            neighbor_targets = [target for _, target in k_neighbors]
            neighbor_distances = [dist for dist, _ in k_neighbors]
            
            # Make prediction based on weights
            if self.weights == 'uniform':
                # Simple average of k nearest neighbors
                prediction = np.mean(neighbor_targets)
            elif self.weights == 'distance':
                # Weighted average (closer neighbors have more influence)
                if any(dist == 0 for dist in neighbor_distances):
                    # Handle exact match - use the exact match value
                    exact_match_idx = neighbor_distances.index(0)
                    prediction = neighbor_targets[exact_match_idx]
                else:
                    # Calculate weighted average
                    weights = [1.0 / dist for dist in neighbor_distances]
                    weighted_sum = sum(w * target for w, target in zip(weights, neighbor_targets))
                    total_weight = sum(weights)
                    prediction = weighted_sum / total_weight
            else:
                raise ValueError(f"Unknown weights: {self.weights}")
            
            predictions.append(prediction)
        
        if self.verbose:
            print("Predictions completed!")
        
        return np.array(predictions)

    def fit_predict(self, X, y):
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
        return self.fit(X, y).predict(X)

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
        return r2_score(y, predictions)

    def get_params(self):
        """Get parameters for this estimator"""
        return {
            'k': self.k,
            'distance_metric': self.distance_metric,
            'weights': self.weights,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self


# ============================================================================
# DEMONSTRATION: Basic KNN Regressor Usage
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATION: Basic KNN Regressor Usage")
print("="*60)

# Create and train the model
print("\n1. Creating and training KNN Regressor:")
knn_regressor = KNNRegressor(k=3, distance_metric='euclidean', weights='uniform', verbose=True)

# Train the model
knn_regressor.fit(X_train, y_train)

# Make predictions
print(f"\n2. Making predictions on test data:")
predictions = knn_regressor.predict(X_test)
print(f"   - Test set size: {len(X_test)}")
print(f"   - First 10 predictions: {predictions[:10]}")
print(f"   - First 10 true values: {y_test[:10]}")

# Calculate R² score
r2 = knn_regressor.score(X_test, y_test)
print(f"\n3. Model evaluation:")
print(f"   - R² Score: {r2:.4f}")

# Test different distance metrics
print(f"\n4. Testing different distance metrics:")
distance_metrics = ['euclidean', 'manhattan']
for metric in distance_metrics:
    knn_metric = KNNRegressor(k=3, distance_metric=metric, verbose=False)
    knn_metric.fit(X_train, y_train)
    r2_metric = knn_metric.score(X_test, y_test)
    print(f"   - {metric.capitalize()} distance: R²={r2_metric:.4f}")

# Test different k values
print(f"\n5. Testing different k values:")
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    knn_k = KNNRegressor(k=k, verbose=False)
    knn_k.fit(X_train, y_train)
    r2_k = knn_k.score(X_test, y_test)
    print(f"   - k={k}: R²={r2_k:.4f}")

# Test different weight schemes
print(f"\n6. Testing different weight schemes:")
weight_schemes = ['uniform', 'distance']
for weights in weight_schemes:
    knn_weights = KNNRegressor(k=3, weights=weights, verbose=False)
    knn_weights.fit(X_train, y_train)
    r2_weights = knn_weights.score(X_test, y_test)
    print(f"   - {weights} weights: R²={r2_weights:.4f}")


# ============================================================================
# DEMONSTRATION: Advanced KNN Regressor Features
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATION: Advanced KNN Regressor Features")
print("="*60)

# Fit-predict demonstration
print(f"\n1. Fit-predict demonstration:")
knn_fit_predict = KNNRegressor(k=3, verbose=False)
train_predictions = knn_fit_predict.fit_predict(X_train, y_train)
train_r2 = r2_score(y_train, train_predictions)
print(f"   - Training R² Score: {train_r2:.4f}")

# Parameter get/set demonstration
print(f"\n2. Parameter management:")
print(f"   - Current parameters: {knn_regressor.get_params()}")
knn_regressor.set_params(k=5, weights='distance')
print(f"   - Updated parameters: {knn_regressor.get_params()}")
knn_regressor.fit(X_train, y_train)
new_r2 = knn_regressor.score(X_test, y_test)
print(f"   - R² Score with new parameters: {new_r2:.4f}")


# ============================================================================
# DEMONSTRATION: Distance Function Comparison
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATION: Distance Function Comparison")
print("="*60)

# Test different distance implementations
test_point1 = np.array([1, 2])
test_point2 = np.array([4, 6])

print(f"\n1. Distance function comparison:")
print(f"   Test points: {test_point1} vs {test_point2}")

# Euclidean distance
euclidean_dist = KNNRegressor.euclidean_distance(test_point1, test_point2)
print(f"   - Euclidean distance: {euclidean_dist:.4f}")

# Manhattan distance
manhattan_dist = KNNRegressor.manhattan_distance(test_point1, test_point2)
print(f"   - Manhattan distance: {manhattan_dist:.4f}")

# Minkowski distance
minkowski_dist = KNNRegressor.minkowski_distance(test_point1, test_point2, p=2)
print(f"   - Minkowski distance (p=2): {minkowski_dist:.4f}")

# Vectorized operations
print(f"\n2. Vectorized distance computation:")
test_points1 = np.array([[1, 2], [3, 4]])
test_points2 = np.array([[4, 6], [7, 8]])

euclidean_vectorized = KNNRegressor.euclidean_distance_vectorized(test_points1, test_points2)
manhattan_vectorized = KNNRegressor.manhattan_distance_vectorized(test_points1, test_points2)

print(f"   - Euclidean vectorized: {euclidean_vectorized}")
print(f"   - Manhattan vectorized: {manhattan_vectorized}")


# ============================================================================
# INTERVIEW QUESTIONS & ANSWERS
# ============================================================================

print("\n" + "="*60)
print("INTERVIEW QUESTIONS & ANSWERS")
print("="*60)

print("\n1. BASIC CONCEPT QUESTIONS:")
print("Q: What is K-Nearest Neighbors Regressor?")
print("A: KNN Regressor is a non-parametric, lazy learning algorithm that predicts")
print("   continuous target values based on the average of k nearest neighbors.")

print("\nQ: How does KNN Regressor differ from KNN Classifier?")
print("A: Regressor predicts continuous values using mean/weighted mean, while")
print("   Classifier predicts discrete classes using majority voting.")

print("\nQ: What are the main parameters of KNN Regressor?")
print("A: k (number of neighbors), distance metric, and weight function.")

print("\n2. ALGORITHM QUESTIONS:")
print("Q: How does KNN Regressor make predictions?")
print("A: 1) Calculate distances to all training points 2) Find k nearest neighbors")
print("   3) Average (uniform) or weighted average (distance) of target values")

print("\nQ: What distance metrics can you use?")
print("A: Euclidean (L2), Manhattan (L1), Minkowski (general), Hamming for categorical")

print("\nQ: How do you handle ties in distance?")
print("A: Use distance-weighted averaging or choose the closest match")

print("\n3. PREDICTION QUESTIONS:")
print("Q: How do you calculate uniform weighted predictions?")
print("A: Simple average: prediction = (1/k) * Σ(target_values)")

print("\nQ: How do you calculate distance weighted predictions?")
print("A: Weighted average: prediction = Σ(wi * target_i) / Σ(wi), where wi = 1/distance_i")

print("\nQ: What happens when distance is zero (exact match)?")
print("A: Use the exact match target value directly")

print("\n4. PARAMETER SELECTION QUESTIONS:")
print("Q: How do you choose the right k value for regression?")
print("A: Use cross-validation, try different k values, consider k=sqrt(n)")

print("\nQ: What happens with k=1 vs k=n?")
print("A: k=1: Overfitting, noisy predictions. k=n: Underfitting, predicts global mean")

print("\nQ: How do you choose distance metrics for regression?")
print("A: Euclidean for continuous features, Manhattan for robust predictions")

print("\n5. PERFORMANCE QUESTIONS:")
print("Q: What's the time complexity of KNN Regressor?")
print("A: Training: O(1), Prediction: O(n*d) where n=samples, d=features")

print("\nQ: How can you make KNN Regressor faster?")
print("A: Use data structures like KD-trees, Ball trees, or approximate nearest neighbors")

print("\nQ: How does KNN Regressor scale with data size?")
print("A: Poorly - prediction time increases linearly with training set size")

print("\n6. EVALUATION QUESTIONS:")
print("Q: How do you evaluate KNN Regressor performance?")
print("A: Use R² score, MSE, MAE, or RMSE")

print("\nQ: What is R² score?")
print("A: R² = 1 - (SS_res / SS_tot), measures proportion of variance explained")

print("\nQ: How do you avoid overfitting in KNN Regressor?")
print("A: Use larger k values, cross-validation, feature selection")

print("\n7. PREPROCESSING QUESTIONS:")
print("Q: Do you need to normalize features for KNN Regressor?")
print("A: Yes! Distance-based algorithms are sensitive to feature scales")

print("\nQ: How do you handle categorical features?")
print("A: Use one-hot encoding, label encoding, or distance metrics like Hamming")

print("\nQ: What about missing values?")
print("A: Impute missing values or use distance metrics that handle missing data")

print("\n8. ADVANTAGES & DISADVANTAGES:")
print("Q: What are the advantages of KNN Regressor?")
print("A: Simple, no assumptions about data distribution, handles non-linear relationships")

print("\nQ: What are the disadvantages?")
print("A: Computationally expensive, sensitive to irrelevant features, curse of dimensionality")

print("\nQ: When would you use KNN Regressor?")
print("A: Small datasets, non-linear relationships, when interpretability is important")

print("\n9. IMPLEMENTATION QUESTIONS:")
print("Q: How would you implement KNN Regressor from scratch?")
print("A: Store training data, implement distance functions, find k nearest, average")

print("\nQ: How would you handle high-dimensional data?")
print("A: Use dimensionality reduction, feature selection, or approximate methods")

print("\nQ: How would you implement weighted KNN Regressor?")
print("A: Weight predictions by inverse distance: weight = 1/distance")

print("\n10. OPTIMIZATION QUESTIONS:")
print("Q: How can you optimize KNN Regressor for speed?")
print("A: Use vectorized operations, data structures (KD-tree), approximate algorithms")

print("\nQ: How would you implement online KNN Regressor?")
print("A: Update distance calculations incrementally as new data arrives")

print("\nQ: How would you handle streaming data?")
print("A: Use sliding window, online learning, or approximate nearest neighbors")

print("\n11. REAL-WORLD QUESTIONS:")
print("Q: How would you use KNN Regressor for price prediction?")
print("A: Use similar products' prices as features, find k nearest similar products")

print("\nQ: How would you apply KNN Regressor to time series?")
print("A: Use lagged values as features, find k nearest similar patterns")

print("\nQ: How would you handle multi-output regression with KNN?")
print("A: Apply KNN independently to each output or use multi-output distance metrics")

print("\n12. DEBUGGING QUESTIONS:")
print("Q: What if KNN Regressor gives poor results?")
print("A: Check feature scaling, try different k values, examine distance metrics")

print("\nQ: How would you debug distance calculations?")
print("A: Test with known distances, visualize nearest neighbors, check for bugs")

print("\nQ: What if predictions are too slow?")
print("A: Reduce training set size, use faster data structures, optimize distance calculations")

print("\n13. MATHEMATICAL QUESTIONS:")
print("Q: What is the mathematical formula for uniform weighted prediction?")
print("A: ŷ = (1/k) * Σ(y_i) for i in k nearest neighbors")

print("\nQ: What is the mathematical formula for distance weighted prediction?")
print("A: ŷ = Σ(w_i * y_i) / Σ(w_i), where w_i = 1/d_i")

print("\nQ: How do you handle the case when distance is zero?")
print("A: Set weight to infinity or use the exact match value directly")

print("\n✅ KNN Regressor Interview Q&A section completed!")

print("\n" + "="*60)
print("QUICK REFERENCE")
print("="*60)

print("\nKey Formulas:")
print("- Euclidean Distance: √(Σ(xi - yi)²)")
print("- Manhattan Distance: Σ|xi - yi|")
print("- Uniform Prediction: (1/k) * Σ(target_values)")
print("- Distance Weighted: Σ(wi * target_i) / Σ(wi), wi = 1/distance_i")

print("\nCommon k Values:")
print("- k=1: Most sensitive to noise")
print("- k=3,5,7: Common values for regression")
print("- k=√n: Rule of thumb")
print("- k=n: Predicts global mean")

print("\nDistance Metrics:")
print("- Euclidean: Default, good for continuous features")
print("- Manhattan: Robust to outliers, good for sparse data")
print("- Hamming: For categorical/binary features")

print("\nWeight Schemes:")
print("- Uniform: Each neighbor has equal influence")
print("- Distance: Closer neighbors have more influence")

print("\nEvaluation Metrics:")
print("- R² Score: Coefficient of determination")
print("- MSE: Mean Squared Error")
print("- MAE: Mean Absolute Error")
print("- RMSE: Root Mean Squared Error")

print("\n✅ KNN Regressor Comprehensive Guide Completed!")
