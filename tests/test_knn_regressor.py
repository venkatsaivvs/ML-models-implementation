from src.ML_code.knn_regressor import KNNRegressor
import numpy as np
import pytest

def test_knn_regressor():
    """Test basic KNN Regressor functionality"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    knn = KNNRegressor(k=3)
    knn.fit(X, y)
    
    assert knn.X_train_ is not None
    assert knn.y_train_ is not None
    assert knn.is_fitted_ == True

def test_distance_functions():
    """Test distance calculation functions"""
    point1 = np.array([1, 2])
    point2 = np.array([4, 6])

    # Euclidean
    euclidean_dist = KNNRegressor.euclidean_distance(point1, point2)
    assert euclidean_dist == 5.0

    # Manhattan
    manhattan_dist = KNNRegressor.manhattan_distance(point1, point2)
    assert manhattan_dist == 7.0

    # Minkowski (p=2 should be Euclidean)
    minkowski_dist = KNNRegressor.minkowski_distance(point1, point2, p=2)
    assert abs(minkowski_dist - 5.0) < 1e-9

def test_unfitted_model_errors():
    """Test that unfitted model raises appropriate errors"""
    knn = KNNRegressor(k=3)
    X = np.array([[1, 1], [2, 2]])
    y = np.array([10, 20])

    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        knn.predict(X)
    
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        knn.score(X, y)

def test_predict_consistency():
    """Test prediction consistency"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    knn = KNNRegressor(k=3, random_state=42)
    knn.fit(X, y)
    
    # Test on training data
    predictions = knn.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    # Test on new points
    new_points = np.array([[1.5, 1.5], [8.5, 8.5]])
    new_predictions = knn.predict(new_points)
    assert len(new_predictions) == 2
    assert all(isinstance(pred, (int, float)) for pred in new_predictions)

def test_different_distance_metrics():
    """Test different distance metrics"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    # Test Euclidean
    knn_euclidean = KNNRegressor(k=3, distance_metric='euclidean')
    knn_euclidean.fit(X, y)
    predictions_euclidean = knn_euclidean.predict(X)
    
    # Test Manhattan
    knn_manhattan = KNNRegressor(k=3, distance_metric='manhattan')
    knn_manhattan.fit(X, y)
    predictions_manhattan = knn_manhattan.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_euclidean) == len(X)
    assert len(predictions_manhattan) == len(X)

def test_different_k_values():
    """Test different k values"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    k_values = [1, 3, 5, 7]
    for k in k_values:
        knn = KNNRegressor(k=k)
        knn.fit(X, y)
        predictions = knn.predict(X)
        
        assert len(predictions) == len(X)
        assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_different_weight_schemes():
    """Test different weight schemes"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    # Test uniform weights
    knn_uniform = KNNRegressor(k=3, weights='uniform')
    knn_uniform.fit(X, y)
    predictions_uniform = knn_uniform.predict(X)
    
    # Test distance weights
    knn_distance = KNNRegressor(k=3, weights='distance')
    knn_distance.fit(X, y)
    predictions_distance = knn_distance.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_uniform) == len(X)
    assert len(predictions_distance) == len(X)

def test_score_method():
    """Test score method (R²)"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    knn = KNNRegressor(k=3)
    knn.fit(X, y)
    
    r2_score = knn.score(X, y)
    
    # R² score should be between 0 and 1 for good fits
    assert 0 <= r2_score <= 1
    
    # For this well-separated data, R² should be high
    assert r2_score > 0.8

def test_fit_predict():
    """Test fit_predict method"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    knn = KNNRegressor(k=3)
    predictions = knn.fit_predict(X, y)
    
    # Check that model is fitted
    assert knn.is_fitted_ == True
    
    # Check predictions
    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_parameter_management():
    """Test parameter get/set methods"""
    knn = KNNRegressor(k=5, distance_metric='manhattan', weights='distance', verbose=False)
    
    # Test get_params
    params = knn.get_params()
    assert 'k' in params
    assert 'distance_metric' in params
    assert 'weights' in params
    assert params['k'] == 5
    assert params['distance_metric'] == 'manhattan'
    assert params['weights'] == 'distance'
    
    # Test set_params
    result = knn.set_params(k=10, distance_metric='euclidean')
    assert result is knn  # Should return self
    assert knn.k == 10
    assert knn.distance_metric == 'euclidean'

def test_edge_cases():
    """Test edge cases"""
    # Test with single sample
    X_single = np.array([[1, 1]])
    y_single = np.array([10])
    knn_single = KNNRegressor(k=1)
    knn_single.fit(X_single, y_single)
    predictions_single = knn_single.predict(X_single)
    
    assert predictions_single.shape == (1,)
    assert predictions_single[0] == 10  # Should predict the only target value
    
    # Test with identical samples
    X_identical = np.array([[1, 1], [1, 1], [1, 1]])
    y_identical = np.array([10, 10, 10])
    knn_identical = KNNRegressor(k=3)
    knn_identical.fit(X_identical, y_identical)
    predictions_identical = knn_identical.predict(X_identical)
    
    assert len(predictions_identical) == 3
    assert all(pred == 10 for pred in predictions_identical)
    
    # Test with very small dataset
    X_small = np.array([[1], [2]])
    y_small = np.array([10, 20])
    knn_small = KNNRegressor(k=1)
    knn_small.fit(X_small, y_small)
    predictions_small = knn_small.predict(X_small)
    
    assert len(predictions_small) == 2

def test_uniform_weighted_prediction():
    """Test uniform weighted prediction"""
    # Simple test case
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30])
    
    knn = KNNRegressor(k=3, weights='uniform')
    knn.fit(X, y)
    
    # Predict for point at x=2 (should be average of all points)
    prediction = knn.predict(np.array([[2]]))[0]
    expected = np.mean(y)  # Should be 20
    
    assert abs(prediction - expected) < 1e-9

def test_distance_weighted_prediction():
    """Test distance weighted prediction"""
    # Test case with clear distance differences
    X = np.array([[1], [2], [10]])  # Point 2 is closest to 1
    y = np.array([10, 20, 100])
    
    knn = KNNRegressor(k=3, weights='distance')
    knn.fit(X, y)
    
    # Predict for point at x=1.5 (closer to points 1 and 2)
    prediction = knn.predict(np.array([[1.5]]))[0]
    
    # Should be closer to 10 and 20 than to 100
    assert 10 <= prediction <= 20

def test_exact_match_handling():
    """Test handling of exact matches in distance weighted prediction"""
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30])
    
    knn = KNNRegressor(k=3, weights='distance')
    knn.fit(X, y)
    
    # Predict for exact match (x=2)
    prediction = knn.predict(np.array([[2]]))[0]
    
    # Should predict exactly 20 (the exact match value)
    assert prediction == 20

def test_regression_accuracy():
    """Test regression accuracy on simple linear relationship"""
    # Create simple linear relationship
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # y = 2x
    
    knn = KNNRegressor(k=3)
    knn.fit(X, y)
    
    # Test predictions
    predictions = knn.predict(X)
    
    # For perfect linear relationship, should fit well
    assert len(predictions) == len(X)
    
    # Test R² score
    r2 = knn.score(X, y)
    assert r2 >= 0  # Should be non-negative

def test_vectorized_distance_functions():
    """Test vectorized distance functions"""
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[4, 6], [7, 8]])
    
    # Test Euclidean vectorized
    euclidean_distances = KNNRegressor.euclidean_distance_vectorized(X1, X2)
    assert len(euclidean_distances) == 2
    
    # Test Manhattan vectorized
    manhattan_distances = KNNRegressor.manhattan_distance_vectorized(X1, X2)
    assert len(manhattan_distances) == 2
    
    # Test that vectorized results match individual calculations
    for i in range(len(X1)):
        euclidean_individual = KNNRegressor.euclidean_distance(X1[i], X2[i])
        manhattan_individual = KNNRegressor.manhattan_distance(X1[i], X2[i])
        
        assert abs(euclidean_distances[i] - euclidean_individual) < 1e-9
        assert abs(manhattan_distances[i] - manhattan_individual) < 1e-9

def test_different_k_values_impact():
    """Test how different k values impact predictions"""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    
    # Test k=1 (should be very sensitive)
    knn_k1 = KNNRegressor(k=1)
    knn_k1.fit(X, y)
    predictions_k1 = knn_k1.predict(X)
    
    # Test k=5 (should be smoother)
    knn_k5 = KNNRegressor(k=5)
    knn_k5.fit(X, y)
    predictions_k5 = knn_k5.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_k1) == len(X)
    assert len(predictions_k5) == len(X)
    
    # k=1 should be more sensitive to local variations
    # k=5 should be smoother
    assert all(isinstance(pred, (int, float)) for pred in predictions_k1)
    assert all(isinstance(pred, (int, float)) for pred in predictions_k5)
