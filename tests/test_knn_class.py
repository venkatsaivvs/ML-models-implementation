import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'ML_code'))

from src.ml_code.knn_search import KNN
import numpy as np
import pytest

def test_knn():
    """Test basic KNN functionality"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    knn = KNN(k=3)
    knn.fit(X, y)
    
    assert knn.X_train_ is not None
    assert knn.y_train_ is not None
    assert knn.is_fitted_ == True

def test_distance_functions():
    """Test distance calculation functions"""
    point1 = np.array([1, 2])
    point2 = np.array([4, 6])
    
    # Test Euclidean distance
    euclidean_dist = KNN.euclidean_distance(point1, point2)
    expected_euclidean = np.sqrt((1-4)**2 + (2-6)**2)  # sqrt(9 + 16) = 5
    assert abs(euclidean_dist - expected_euclidean) < 1e-10
    
    # Test Manhattan distance
    manhattan_dist = KNN.manhattan_distance(point1, point2)
    expected_manhattan = abs(1-4) + abs(2-6)  # 3 + 4 = 7
    assert manhattan_dist == expected_manhattan

def test_unfitted_model_errors():
    """Test that unfitted model raises appropriate errors"""
    knn = KNN(k=3)
    X = np.array([[1, 1], [2, 2]])
    y = np.array([0, 1])
    
    # All methods should raise errors before fitting
    with pytest.raises(ValueError, match="Model must be fitted"):
        knn.predict(X)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        knn.predict_proba(X)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        knn.score(X, y)

def test_predict_consistency():
    """Test prediction consistency"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    knn = KNN(k=3, random_state=42)
    knn.fit(X, y)
    
    # Test on training data
    predictions = knn.predict(X)
    assert len(predictions) == len(X)
    
    # Test on new points - should be clearly classified
    new_points = np.array([[1.5, 1.5], [8.5, 8.5]])
    new_predictions = knn.predict(new_points)
    assert len(new_predictions) == 2
    assert all(pred in [0, 1] for pred in new_predictions)

def test_different_weights():
    """Test different weight schemes"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Test uniform weights
    knn_uniform = KNN(k=3, weights='uniform')
    knn_uniform.fit(X, y)
    predictions_uniform = knn_uniform.predict(X)
    
    # Test distance weights
    knn_distance = KNN(k=3, weights='distance')
    knn_distance.fit(X, y)
    predictions_distance = knn_distance.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_uniform) == len(X)
    assert len(predictions_distance) == len(X)

def test_different_k_values():
    """Test different k values"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    k_values = [1, 3, 5]
    for k in k_values:
        knn = KNN(k=k)
        knn.fit(X, y)
        predictions = knn.predict(X)
        assert len(predictions) == len(X)

def test_predict_proba():
    """Test probability predictions"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    knn = KNN(k=3)
    knn.fit(X, y)
    
    probabilities = knn.predict_proba(X)
    
    # Check shape
    assert probabilities.shape == (len(X), 2)  # 2 classes
    
    # Check that probabilities sum to 1
    np.testing.assert_array_almost_equal(
        np.sum(probabilities, axis=1), 
        np.ones(len(X))
    )
    
    # Check that all probabilities are non-negative
    assert np.all(probabilities >= 0)

def test_score_method():
    """Test score method"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    knn = KNN(k=3)
    knn.fit(X, y)
    
    score = knn.score(X, y)
    
    # Score should be between 0 and 1
    assert 0 <= score <= 1
    
    # For this well-separated data, score should be high
    assert score > 0.8

def test_fit_predict():
    """Test fit_predict method"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    knn = KNN(k=3)
    predictions = knn.fit_predict(X, y)
    
    # Check that model is fitted
    assert knn.is_fitted_ == True
    
    # Check predictions
    assert len(predictions) == len(X)
    assert all(pred in [0, 1] for pred in predictions)