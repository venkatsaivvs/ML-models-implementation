"""
K-Nearest Neighbors (KNN) - Step by Step Learning
is a lazy learning algorithm that classifies data points based on the majority class of their k nearest neighbors in feature space.
meaning it doesn't learn a model during training. It stores all training data and computes distances only during prediction.
weighted voting is used to predict the class of a data point.
list all functions and classes in the file
1. euclidean_distance
2. manhattan_distance
3. minkowski_distance

6. _compute_distance
7. 
8. fit
9. predict
10. predict_proba
11. fit_predict
12. score
13. get_params
14. set_params
=========================================
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate sample data for demonstration
np.random.seed(42)

# Create classification dataset
X_class, y_class = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1, random_state=42)

# Create blob dataset for visualization
X_blobs, y_blobs = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)


class KNN:
    """
    K-Nearest Neighbors Classifier with comprehensive functionality
    Following scikit-learn style interface for interview preparation
    """
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform', verbose=False):
        """
        Initialize KNN classifier
        
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
            print(f"Initialized KNN with k={k}, distance_metric='{distance_metric}', weights='{weights}'")
    
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

        X = np.array(X)
        y = np.array(y)
        
        if self.verbose:
            print(f"Training KNN with {len(X)} samples, {X.shape[1]} features...")
            print(f"Number of unique classes: {len(np.unique(y))}")
        
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.is_fitted_ = True
        
        if self.verbose:
            print("Training completed! Model is ready for predictions.")
        
        return self

    def predict(self, X):

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.array(X)
        predictions = []
        
        if self.verbose:
            print(f"Making predictions for {len(X)} test samples...")
        
        for test_point in X:
            # Compute distances to all training points
            distances = {}
            
            for i, train_point in enumerate(self.X_train_):
                dist = self._compute_distance(test_point, train_point)
                distances[i] = dist
            
            # Sort by distance and get k nearest neighbors
            sorted_dict = dict(sorted(distances.items(), key=lambda item: item[1]))
            neighbor_indices = list(sorted_dict.keys())[:self.k]
            
            # Extract labels and distances
            neighbor_labels = [self.y_train_[idx] for idx in neighbor_indices]
            neighbor_distances = [sorted_dict[idx] for idx in neighbor_indices]
            
            # Make prediction based on weights
            if self.weights == 'uniform':
                # Simple majority vote
                prediction = Counter(neighbor_labels).most_common(1)[0][0]
            elif self.weights == 'distance':
                # Weighted vote (closer neighbors have more influence)
                weighted_votes = {}
                for label, distance in zip(neighbor_labels, neighbor_distances):
                    if distance == 0:  # Handle exact match
                        weight = float('inf')
                    else:
                        weight = 1.0 / distance
                    
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                
                prediction = max(weighted_votes, key=weighted_votes.get)
            else:
                raise ValueError(f"Unknown weights: {self.weights}")
            
            predictions.append(prediction)
        
        if self.verbose:
            print("Predictions completed!")
        
        return np.array(predictions)

    def predict_proba(self, X):

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.array(X)
        unique_classes = np.unique(self.y_train_)
        n_classes = len(unique_classes)
        probabilities = []
        
        if self.verbose:
            print(f"Computing probabilities for {len(X)} test samples...")
        
        for x in X:
            # Compute distances to all training points
            distances = {}
            for i, x_train in enumerate(self.X_train_):
                dist = self._compute_distance(x, x_train)
                distances[i] = dist
            
            # Sort by distance and get k nearest neighbors
            sorted_dict = dict(sorted(distances.items(), key=lambda item: item[1]))
            neighbor_indices = list(sorted_dict.keys())[:self.k]
            
            # Extract labels and distances
            neighbor_labels = [self.y_train_[idx] for idx in neighbor_indices]
            neighbor_distances = [sorted_dict[idx] for idx in neighbor_indices]
            
            # Compute probabilities based on weights
            class_probs = np.zeros(n_classes)
            
            if self.weights == 'uniform':
                # Count occurrences and normalize
                for label in neighbor_labels:
                    class_idx = np.where(unique_classes == label)[0][0]
                    class_probs[class_idx] += 1
                class_probs = class_probs / self.k
                
            elif self.weights == 'distance':
                # Weighted probabilities
                total_weight = 0
                for label, distance in zip(neighbor_labels, neighbor_distances):
                    if distance == 0:  # Handle exact match
                        weight = float('inf')
                    else:
                        weight = 1.0 / distance
                    
                    class_idx = np.where(unique_classes == label)[0][0]
                    class_probs[class_idx] += weight
                    total_weight += weight
                
                class_probs = class_probs / total_weight
            
            probabilities.append(class_probs)
        
        if self.verbose:
            print("Probability computation completed!")
        
        return np.array(probabilities)

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        #recall score
        #recall = recall_score(y, predictions)
        #return recall
        return accuracy_score(y, predictions)

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
# DEMONSTRATION: Basic KNN Usage
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATION: Basic KNN Usage")
print("="*60)

# Create and train the model
print("\n1. Creating and training KNN model:")
knn_model = KNN(k=3, distance_metric='euclidean', weights='uniform', verbose=True)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions
print(f"\n2. Making predictions on test data:")
predictions = knn_model.predict(X_test)
print(f"   - Test set size: {len(X_test)}")
print(f"   - First 10 predictions: {predictions[:10]}")
print(f"   - First 10 true labels: {y_test[:10]}")

# Calculate accuracy
accuracy = knn_model.score(X_test, y_test)
print(f"\n3. Model evaluation:")
print(f"   - Accuracy: {accuracy:.4f}")

# Test different distance metrics
print(f"\n4. Testing different distance metrics:")
distance_metrics = ['euclidean', 'manhattan']
for metric in distance_metrics:
    knn_metric = KNN(k=3, distance_metric=metric, verbose=False)
    knn_metric.fit(X_train, y_train)
    acc = knn_metric.score(X_test, y_test)
    print(f"   - {metric.capitalize()} distance: {acc:.4f}")

# Test different k values
print(f"\n5. Testing different k values:")
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    knn_k = KNN(k=k, verbose=False)
    knn_k.fit(X_train, y_train)
    acc = knn_k.score(X_test, y_test)
    print(f"   - k={k}: {acc:.4f}")

# Test different weight schemes
print(f"\n6. Testing different weight schemes:")
weight_schemes = ['uniform', 'distance']
for weights in weight_schemes:
    knn_weights = KNN(k=3, weights=weights, verbose=False)
    knn_weights.fit(X_train, y_train)
    acc = knn_weights.score(X_test, y_test)
    print(f"   - {weights} weights: {acc:.4f}")


# ============================================================================
# DEMONSTRATION: Advanced KNN Features
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATION: Advanced KNN Features")
print("="*60)

# Probability predictions
print("\n1. Probability predictions:")
knn_proba = KNN(k=5, verbose=False)
knn_proba.fit(X_train, y_train)

# Get probabilities for first 5 test samples
probabilities = knn_proba.predict_proba(X_test[:5])
print(f"   - Probabilities for first 5 test samples:")
for i, prob in enumerate(probabilities):
    print(f"     Sample {i}: {prob}")

# Fit-predict demonstration
print(f"\n2. Fit-predict demonstration:")
knn_fit_predict = KNN(k=3, verbose=False)
train_predictions = knn_fit_predict.fit_predict(X_train, y_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"   - Training accuracy: {train_accuracy:.4f}")

# Parameter get/set demonstration
print(f"\n3. Parameter management:")
print(f"   - Current parameters: {knn_model.get_params()}")
knn_model.set_params(k=5, weights='distance')
print(f"   - Updated parameters: {knn_model.get_params()}")
knn_model.fit(X_train, y_train)
new_accuracy = knn_model.score(X_test, y_test)
print(f"   - Accuracy with new parameters: {new_accuracy:.4f}")


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
euclidean_dist = KNN.euclidean_distance(test_point1, test_point2)
print(f"   - Euclidean distance: {euclidean_dist:.4f}")

# Manhattan distance
manhattan_dist = KNN.manhattan_distance(test_point1, test_point2)
print(f"   - Manhattan distance: {manhattan_dist:.4f}")

# Minkowski distance
minkowski_dist = KNN.minkowski_distance(test_point1, test_point2, p=2)
print(f"   - Minkowski distance (p=2): {minkowski_dist:.4f}")

# Vectorized operations
print(f"\n2. Vectorized distance computation:")
test_points1 = np.array([[1, 2], [3, 4]])
test_points2 = np.array([[4, 6], [7, 8]])

euclidean_vectorized = KNN.euclidean_distance_vectorized(test_points1, test_points2)
manhattan_vectorized = KNN.manhattan_distance_vectorized(test_points1, test_points2)

print(f"   - Euclidean vectorized: {euclidean_vectorized}")
print(f"   - Manhattan vectorized: {manhattan_vectorized}")


# ============================================================================
# INTERVIEW QUESTIONS & ANSWERS
# ============================================================================

print("\n" + "="*60)
print("INTERVIEW QUESTIONS & ANSWERS")
print("="*60)

print("\n1. BASIC CONCEPT QUESTIONS:")
print("Q: What is K-Nearest Neighbors?")
print("A: KNN is a non-parametric, lazy learning algorithm that classifies data points")
print("   based on the majority class of their k nearest neighbors in feature space.")

print("\nQ: Why is KNN called a 'lazy' algorithm?")
print("A: KNN doesn't learn a model during training. It stores all training data and")
print("   computes distances only during prediction, making it lazy.")

print("\nQ: What are the main parameters of KNN?")
print("A: k (number of neighbors), distance metric, and weight function.")

print("\n2. ALGORITHM QUESTIONS:")
print("Q: How does KNN make predictions?")
print("A: 1) Calculate distances to all training points 2) Find k nearest neighbors")
print("   3) Vote (uniform) or weighted vote (distance) for prediction")

print("\nQ: What distance metrics can you use?")
print("A: Euclidean (L2), Manhattan (L1), Minkowski (general), Hamming for categorical")

print("\nQ: How do you handle ties in voting?")
print("A: Use distance-weighted voting or choose the class with smaller index")

print("\n3. PARAMETER SELECTION QUESTIONS:")
print("Q: How do you choose the right k value?")
print("A: Use cross-validation, try odd k values to avoid ties, consider k=sqrt(n)")

print("\nQ: What happens with k=1 vs k=n?")
print("A: k=1: Overfitting, sensitive to noise. k=n: Underfitting, predicts majority class")

print("\nQ: How do you choose distance metrics?")
print("A: Euclidean for continuous features, Manhattan for sparse data, Hamming for categorical")

print("\n4. PERFORMANCE QUESTIONS:")
print("Q: What's the time complexity of KNN?")
print("A: Training: O(1), Prediction: O(n*d) where n=samples, d=features")

print("\nQ: How can you make KNN faster?")
print("A: Use data structures like KD-trees, Ball trees, or approximate nearest neighbors")

print("\nQ: How does KNN scale with data size?")
print("A: Poorly - prediction time increases linearly with training set size")

print("\n5. PREPROCESSING QUESTIONS:")
print("Q: Do you need to normalize features for KNN?")
print("A: Yes! Distance-based algorithms are sensitive to feature scales")

print("\nQ: How do you handle categorical features?")
print("A: Use one-hot encoding, label encoding, or distance metrics like Hamming")

print("\nQ: What about missing values?")
print("A: Impute missing values or use distance metrics that handle missing data")

print("\n6. ADVANTAGES & DISADVANTAGES:")
print("Q: What are the advantages of KNN?")
print("A: Simple, no assumptions about data distribution, works for any number of classes")

print("\nQ: What are the disadvantages?")
print("A: Computationally expensive, sensitive to irrelevant features, curse of dimensionality")

print("\nQ: When would you use KNN?")
print("A: Small datasets, non-linear boundaries, when interpretability is important")

print("\n7. IMPLEMENTATION QUESTIONS:")
print("Q: How would you implement KNN from scratch?")
print("A: Store training data, implement distance functions, find k nearest, vote")

print("\nQ: How would you handle high-dimensional data?")
print("A: Use dimensionality reduction, feature selection, or approximate methods")

print("\nQ: How would you implement weighted KNN?")
print("A: Weight votes by inverse distance: weight = 1/distance")

print("\n8. OPTIMIZATION QUESTIONS:")
print("Q: How can you optimize KNN for speed?")
print("A: Use vectorized operations, data structures (KD-tree), approximate algorithms")

print("\nQ: How would you implement online KNN?")
print("A: Update distance calculations incrementally as new data arrives")

print("\nQ: How would you handle streaming data?")
print("A: Use sliding window, online learning, or approximate nearest neighbors")

print("\n9. EVALUATION QUESTIONS:")
print("Q: How do you evaluate KNN performance?")
print("A: Use cross-validation, accuracy, precision, recall, F1-score")

print("\nQ: How do you avoid overfitting in KNN?")
print("A: Use larger k values, cross-validation, feature selection")

print("\nQ: How would you validate your KNN implementation?")
print("A: Compare with sklearn implementation, test on known datasets")

print("\n10. ADVANCED QUESTIONS:")
print("Q: How would you implement KNN for regression?")
print("A: Average (uniform) or weighted average (distance) of k nearest neighbors")

print("\nQ: How would you handle imbalanced datasets?")
print("A: Use weighted voting, SMOTE, or cost-sensitive learning")

print("\nQ: How would you implement KNN for multi-output problems?")
print("A: Apply KNN independently to each output or use multi-output distance metrics")

print("\n11. REAL-WORLD QUESTIONS:")
print("Q: How would you use KNN for recommendation systems?")
print("A: Find k most similar users, recommend items they liked")

print("\nQ: How would you apply KNN to image recognition?")
print("A: Use pixel values as features, or extract features like color histograms")

print("\nQ: How would you handle text classification with KNN?")
print("A: Use TF-IDF, word embeddings, or character n-grams as features")

print("\n12. DEBUGGING QUESTIONS:")
print("Q: What if KNN gives poor results?")
print("A: Check feature scaling, try different k values, examine distance metrics")

print("\nQ: How would you debug distance calculations?")
print("A: Test with known distances, visualize nearest neighbors, check for bugs")

print("\nQ: What if predictions are too slow?")
print("A: Reduce training set size, use faster data structures, optimize distance calculations")

print("\n✅ KNN Interview Q&A section completed!")

print("\n" + "="*60)
print("QUICK REFERENCE")
print("="*60)

print("\nKey Formulas:")
print("- Euclidean Distance: √(Σ(xi - yi)²)")
print("- Manhattan Distance: Σ|xi - yi|")
print("- Minkowski Distance: (Σ|xi - yi|^p)^(1/p)")
print("- Distance Weight: 1/distance")

print("\nCommon k Values:")
print("- k=1: Most sensitive to noise")
print("- k=3,5,7: Common odd values")
print("- k=√n: Rule of thumb")
print("- k=n: Predicts majority class")

print("\nDistance Metrics:")
print("- Euclidean: Default, good for continuous features")
print("- Manhattan: Robust to outliers, good for sparse data")
print("- Hamming: For categorical/binary features")

print("\nWeight Schemes:")
print("- Uniform: Each neighbor has equal vote")
print("- Distance: Closer neighbors have more influence")

print("\n✅ KNN Comprehensive Guide Completed!")