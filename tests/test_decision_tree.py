from src.ML_code.decisiontreeclass import DecisionTree
import numpy as np
import pytest

def test_decision_tree():
    """Test basic Decision Tree functionality"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    
    assert dt.tree_ is not None
    assert dt.classes_ is not None
    assert dt.is_fitted_ == True

def test_impurity_functions():
    """Test impurity calculation functions"""
    # Test with pure labels (should have 0 impurity)
    pure_labels = np.array([1, 1, 1, 1])
    assert DecisionTree.compute_gini(pure_labels) == 0.0
    assert DecisionTree.compute_entropy(pure_labels) == 0.0
    
    # Test with mixed labels
    mixed_labels = np.array([0, 0, 1, 1])
    gini_mixed = DecisionTree.compute_gini(mixed_labels)
    entropy_mixed = DecisionTree.compute_entropy(mixed_labels)
    
    assert gini_mixed > 0
    assert entropy_mixed > 0
    
    # Test with empty array
    assert DecisionTree.compute_gini(np.array([])) == 0.0
    assert DecisionTree.compute_entropy(np.array([])) == 0.0

def test_unfitted_model_errors():
    """Test that unfitted model raises appropriate errors"""
    dt = DecisionTree()
    X = np.array([[1, 1], [2, 2]])
    y = np.array([0, 1])
    
    # All methods should raise errors before fitting
    with pytest.raises(ValueError, match="Model must be fitted"):
        dt.predict(X)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        dt.predict_proba(X)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        dt.score(X, y)

def test_predict_consistency():
    """Test prediction consistency"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    
    # Test on training data
    predictions = dt.predict(X)
    assert len(predictions) == len(X)
    assert all(pred in [0, 1] for pred in predictions)
    
    # Test on new points
    new_points = np.array([[1.5, 1.5], [8.5, 8.5]])
    new_predictions = dt.predict(new_points)
    assert len(new_predictions) == 2
    assert all(pred in [0, 1] for pred in new_predictions)

def test_different_criteria():
    """Test different splitting criteria"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Test Gini criterion
    dt_gini = DecisionTree(max_depth=2, criterion='gini')
    dt_gini.fit(X, y)
    predictions_gini = dt_gini.predict(X)
    
    # Test Entropy criterion
    dt_entropy = DecisionTree(max_depth=2, criterion='entropy')
    dt_entropy.fit(X, y)
    predictions_entropy = dt_entropy.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_gini) == len(X)
    assert len(predictions_entropy) == len(X)

def test_different_max_depths():
    """Test different max depth values"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    max_depths = [1, 2, 3, None]
    for max_depth in max_depths:
        dt = DecisionTree(max_depth=max_depth)
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

def test_predict_proba():
    """Test probability predictions"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    dt = DecisionTree(max_depth=2)
    dt.fit(X, y)
    
    probabilities = dt.predict_proba(X)
    
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
    
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    
    score = dt.score(X, y)
    
    # Score should be between 0 and 1
    assert 0 <= score <= 1
    
    # For this well-separated data, score should be high
    assert score > 0.8

def test_fit_predict():
    """Test fit_predict method"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    dt = DecisionTree(max_depth=2)
    predictions = dt.fit_predict(X, y)
    
    # Check that model is fitted
    assert dt.is_fitted_ == True
    
    # Check predictions
    assert len(predictions) == len(X)
    assert all(pred in [0, 1] for pred in predictions)

def test_tree_properties():
    """Test tree depth and leaf count"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Test with limited depth
    dt = DecisionTree(max_depth=2)
    dt.fit(X, y)
    
    depth = dt.get_depth()
    leaves = dt.get_n_leaves()
    
    assert depth <= 2
    assert leaves >= 1
    assert isinstance(depth, int)
    assert isinstance(leaves, int)

def test_min_samples_constraints():
    """Test minimum samples constraints"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Class 1
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Test with high min_samples_split
    dt = DecisionTree(min_samples_split=5)
    dt.fit(X, y)
    
    # Should still work but might create simpler tree
    predictions = dt.predict(X)
    assert len(predictions) == len(X)
    
    # Test with high min_samples_leaf
    dt_leaf = DecisionTree(min_samples_leaf=3)
    dt_leaf.fit(X, y)
    
    predictions_leaf = dt_leaf.predict(X)
    assert len(predictions_leaf) == len(X)
