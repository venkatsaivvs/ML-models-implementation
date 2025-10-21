"""
K-Means Clustering - Step by Step Learning

#list all functions and classes in the file
1. euclidean_distance
2. euclidean_distance_norm
3. calculate_inertia -sum of sq distances of points to their closest centroid within-cluster sum of squared distances
4. compute_silhouette_score - minimize with in cluster distance and maximize with out cluster distance
silhouette score = (b - a) / max(a, b) where a is the average distance to points in the same cluster and b
b is the minimum average distance to points in other clusters
5. initialize_centroids - randomly initialize centroids
6. assign_clusters - calculate the distance between each point and each centroid and assign the point to the closest centroid
7. update_centroids - update centroids to the mean of the points in the cluster
8. fit
9. predict
10. fit_predict
11. transform
12. score
=========================================
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
                  [8, 8], [8, 9], [9, 8], [9, 9]])

np.random.seed(42)
cluster1 = np.random.normal([2, 2], 0.5, (30, 2))
cluster2 = np.random.normal([6, 6], 0.5, (30, 2))
cluster3 = np.random.normal([2, 6], 0.5, (30, 2))
X = np.vstack([cluster1, cluster2, cluster3])

class KMeans:
    """
    K-Means Clustering Class with fit() and predict() methods
    Following scikit-learn style interface
    """
    
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, distance='euclidean', random_state=None, verbose=False):
        """
        Initialize K-Means Clustering
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence
        distance : str, default='euclidean'
            Distance metric to use ('euclidean', 'manhattan')
        random_state : int, default=None
            Random seed for reproducibility
        verbose : bool, default=False
            Whether to print training information
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance = distance
        self.random_state = random_state
        self.verbose = verbose
        
        # Model state attributes
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
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
    def euclidean_distance_norm(v1, v2):
        return np.linalg.norm(v1 - v2)

    def _compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute distance between two points using specified metric"""
        if self.distance == 'euclidean':
            return self.euclidean_distance(point1, point2)
        elif self.distance == 'manhattan':
            return self.manhattan_distance(point1, point2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

    @staticmethod
    def calculate_inertia(X, centroids, labels):
        """Compute inertia using vectorized operations"""
        diff = X - centroids[labels]
        squared_distances = np.sum(diff ** 2, axis=1)
        return float(np.sum(squared_distances))

    @staticmethod
    def compute_silhouette_score(X, labels):
        n = len(X)
        unique_labels = np.unique(labels)
        silhouette_scores = []

        for i in range(n):
            same_cluster = X[labels == labels[i]]
            other_clusters = [X[labels == l] for l in unique_labels if l != labels[i]]
            # a(i): average distance to points in same cluster
            a = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster if not np.array_equal(x, X[i])] or [0])

            # b(i): min average distance to points in other clusters
            b = np.min([np.mean([np.linalg.norm(X[i] - x) for x in cluster]) for cluster in other_clusters])

            silhouette_scores.append((b - a) / max(a, b))
        
        return np.mean(silhouette_scores)

    def initialize_centroids(self, X, k):
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_indices]
        for i, centroid in enumerate(centroids):
            print(f"     Centroid {i}: {centroid}")
        return centroids

    def assign_clusters(self, X, centroids):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)  # Array to store cluster assignments
        for i in range(n_samples):
            point = X[i]
            distances = []
            for j in range(len(centroids)):
                dist = self._compute_distance(point, centroids[j])
                distances.append(dist)
            closest_centroid = np.argmin(distances)
            labels[i] = closest_centroid
            if i < 3:
                print(f"   Point {i} {point} → Centroid {closest_centroid} (distances: {[f'{d:.2f}' for d in distances]})")
        return labels

    def update_centroids(self, X, labels, k, centroids=None):
        """ Move centroids to the center (mean) of their assigned points """
        print("   Updating centroids to center of their clusters...")
        
        n_features = X.shape[1]
        new_centroids = np.zeros((k, n_features))
        for cluster_id in range(k):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroids[cluster_id] = new_centroid
            else:
                # If cluster is empty, keep the old centroid or initialize randomly
                if centroids is not None:
                    new_centroids[cluster_id] = centroids[cluster_id]
                else:
                    # Random initialization if no previous centroids
                    new_centroids[cluster_id] = np.random.normal(0, 1, n_features)
        return new_centroids
    
    def fit(self, X):

        X = np.array(X)
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"Training K-means with {self.n_clusters} clusters on {n_samples} samples...")
        
        self.centroids_ = self.initialize_centroids(X, self.n_clusters)
        
        # Training loop
        for iteration in range(self.max_iter):
            old_centroids = self.centroids_.copy()
            
            # Assign clusters
            self.labels_ = self.assign_clusters(X, self.centroids_)
            # Update centroids
            self.centroids_ = self.update_centroids(X, self.labels_, self.n_clusters, self.centroids_)
            # Calculate inertia
            self.inertia_ = self.calculate_inertia(X, self.centroids_, self.labels_)
            
            if self.verbose:
                print(f"   Inertia after iteration {iteration + 1}: {self.inertia_:.2f}")
            
            # Check convergence
            moved = False
            for i in range(self.n_clusters):
                distance = self._compute_distance(old_centroids[i], self.centroids_[i])
                if distance > self.tol:
                    moved = True
                    break
            
            if not moved:
                if self.verbose:
                    print(f"\n   🎉 Algorithm converged after {iteration + 1} iterations!")
                self.n_iter_ = iteration + 1
                break
        else:
            # If we didn't break, we reached max_iter
            self.n_iter_ = self.max_iter
            if self.verbose:
                print(f"\n   ⚠️  Algorithm reached maximum iterations ({self.max_iter})")
        
        return self
    
    def predict(self, X):

        if self.centroids_ is None:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            point = X[i]
            distances = []
            for j in range(len(self.centroids_)):
                dist = self._compute_distance(point, self.centroids_[j])
                distances.append(dist)
            closest_centroid = np.argmin(distances)
            labels[i] = closest_centroid
        
        return labels
    
    def fit_predict(self, X):
        return self.fit(X).labels_
    
    def transform(self, X):
        """
        Transform data to cluster-distance spacef
        """
        if self.centroids_ is None:
            raise ValueError("Model must be fitted before transforming. Call fit() first.")
        
        X = np.array(X)
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            point = X[i]
            for j in range(self.n_clusters):
                distances[i, j] = self._compute_distance(point, self.centroids_[j])
        
        return distances
    
    def score(self, X):
        if self.centroids_ is None:
            raise ValueError("Model must be fitted before scoring. Call fit() first.")
        
        labels = self.predict(X)
        inertia = self.calculate_inertia(X, self.centroids_, labels)
        return -inertia  # Negative because higher is better







# ============================================================================
# DEMONSTRATION: Using the new KMeans class with fit() and predict() methods
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATION: KMeans Class with fit() and predict()")
print("="*60)

# Create and train the model
print("\n1. Creating and training KMeans model (Euclidean distance):")
kmeans_model = KMeans(n_clusters=3, max_iter=100, tol=1e-4, distance='euclidean', random_state=42, verbose=True)

# Train the model (fit method)
kmeans_model.fit(X)


predicted_labels = kmeans_model.predict(X)
print(f"   - Predicted labels shape: {predicted_labels.shape}")
print(f"   - First 10 predicted labels: {predicted_labels[:10]}")

# Create some new test data points
print(f"\n4. Testing on new data points:")
new_points = np.array([
    [1.5, 1.5],  # Should be close to cluster 0
    [5.8, 5.8],  # Should be close to cluster 1  
    [2.2, 5.8],  # Should be close to cluster 2
    [4.0, 4.0]   # In between clusters
])

new_predictions = kmeans_model.predict(new_points)
print(f"   - New points: {new_points}")
print(f"   - Predicted clusters: {new_predictions}")




# Calculate model score
print(f"\n7. Model evaluation:")
score = kmeans_model.score(X)
print(f"   - Model score (negative inertia): {score:.2f}")

