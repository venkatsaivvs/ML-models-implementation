from src.ml_code.kmeans import KMeans, calculate_inertia
import numpy as np
import pytest
#tdd test driven development approach

def test_kmeans():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    assert kmeans.centroids_ is not None
    assert kmeans.labels_ is not None
    assert kmeans.inertia_ is not None
    assert kmeans.n_iter_ is not None

def test_convergence_and_inertia():
    # Create well-separated clusters
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Cluster 2
    
    kmeans = KMeans(n_clusters=2, max_iter=100, random_state=42)
    kmeans.fit(X)
    
    # Should converge quickly with well-separated data
    assert kmeans.n_iter_ < 10
    assert kmeans.inertia_ > 0
    assert kmeans.centroids_.shape == (2, 2)
    assert len(np.unique(kmeans.labels_)) == 2


def test_unfitted_model_errors():
    kmeans = KMeans(n_clusters=2)
    X = np.array([[1, 1], [2, 2]])
    
    # All methods should raise errors before fitting
    with pytest.raises(ValueError, match="Model must be fitted"):
        kmeans.predict(X)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        kmeans.transform(X)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        kmeans.score(X)


def test_predict_consistency():
    X = np.array([[1, 1], [1, 2], [8, 8], [8, 9]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Predictions on training data should match fitted labels
    predictions = kmeans.predict(X)
    assert np.array_equal(predictions, kmeans.labels_)
    
    # Test on new points
    new_points = np.array([[1.5, 1.5], [8.5, 8.5]])
    new_predictions = kmeans.predict(new_points)
    assert len(new_predictions) == 2
    assert all(pred in [0, 1] for pred in new_predictions)


def test_inertia_calculation():
    """Test inertia calculation and its properties"""
    # Create well-separated clusters for predictable inertia
    cluster1 = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])  # Tight cluster
    cluster2 = np.array([[8, 8], [8, 9], [9, 8], [9, 9]])  # Tight cluster
    X = np.vstack([cluster1, cluster2])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Test basic inertia properties
    assert kmeans.inertia_ > 0, "Inertia should be positive"
    assert isinstance(kmeans.inertia_, (int, float)), "Inertia should be numeric"
    
    # Test inertia decreases with more clusters (for same data)
    kmeans_3 = KMeans(n_clusters=3, random_state=42)
    kmeans_3.fit(X)
    
    # More clusters should generally have lower inertia (better fit)
    assert kmeans_3.inertia_ <= kmeans.inertia_, "More clusters should have lower or equal inertia"
    
    # Test inertia with perfect clusters (should be very low)
    perfect_clusters = np.array([[0, 0], [0, 1], [1, 0], [1, 1],  # Cluster 1
                                 [10, 10], [10, 11], [11, 10], [11, 11]])  # Cluster 2
    kmeans_perfect = KMeans(n_clusters=2, random_state=42)
    kmeans_perfect.fit(perfect_clusters)
    
    # Perfect clusters should have very low inertia
    assert kmeans_perfect.inertia_ < 1.0, "Perfect clusters should have very low inertia"
    
    # Test inertia consistency - same data, same random_state should give same inertia
    kmeans1 = KMeans(n_clusters=2, random_state=42)
    kmeans2 = KMeans(n_clusters=2, random_state=42)
    kmeans1.fit(X)
    kmeans2.fit(X)
    
    assert abs(kmeans1.inertia_ - kmeans2.inertia_) < 1e-10, "Same random_state should give same inertia"


def test_calculate_inertia_function():
    """Test the calculate_inertia function directly"""
    # Test with simple 2D data
    X = np.array([[1, 1], [2, 2], [3, 3]])
    centroids = np.array([[1.5, 1.5], [2.5, 2.5]])  # 2 centroids
    labels = np.array([0, 0, 1])  # First 2 points to cluster 0, last to cluster 1
    
    inertia = calculate_inertia(X, centroids, labels)
    
    # Test basic properties
    assert inertia > 0, "Inertia should be positive"
    assert isinstance(inertia, (int, float)), "Inertia should be numeric"
    
    # Test with perfect clustering (points exactly at centroids)
    X_perfect = np.array([[1, 1], [2, 2]])
    centroids_perfect = np.array([[1, 1], [2, 2]])
    labels_perfect = np.array([0, 1])
    
    inertia_perfect = calculate_inertia(X_perfect, centroids_perfect, labels_perfect)
    assert inertia_perfect == 0, "Perfect clustering should have zero inertia"
    
    # Test with single point
    X_single = np.array([[1, 1]])
    centroids_single = np.array([[1, 1]])
    labels_single = np.array([0])
    
    inertia_single = calculate_inertia(X_single, centroids_single, labels_single)
    assert inertia_single == 0, "Single point at centroid should have zero inertia"
    
    # Test manual calculation
    # Point [1,1] assigned to centroid [1.5,1.5] -> distance = sqrt((1-1.5)² + (1-1.5)²) = sqrt(0.5)
    # Point [2,2] assigned to centroid [1.5,1.5] -> distance = sqrt((2-1.5)² + (2-1.5)²) = sqrt(0.5)  
    # Point [3,3] assigned to centroid [2.5,2.5] -> distance = sqrt((3-2.5)² + (3-2.5)²) = sqrt(0.5)
    # Total inertia = 0.5 + 0.5 + 0.5 = 1.5
    expected_inertia = 1.5
    assert abs(inertia - expected_inertia) < 1e-10, f"Expected inertia {expected_inertia}, got {inertia}"
    