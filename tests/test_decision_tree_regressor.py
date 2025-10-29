from src.ml_code.decision_tree_regressor import DecisionTreeRegressor
import numpy as np
import pytest

"""list all test functions in the file
1. test_decision_tree_regressor
2. test_impurity_functions
3. test_unfitted_model_errors
4. test_predict_consistency
5. test_different_criteria
6. test_different_max_depths
7. test_score_method
8. test_fit_predict
9. test_tree_properties
10. test_min_samples_constraints
11. test_leaf_values
12. test_parameter_management
13. test_edge_cases
14. test_regression_accuracy
15. test_information_gain
"""

def test_decision_tree_regressor():
    """Test basic Decision Tree Regressor functionality"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    dt = DecisionTreeRegressor(max_depth=3)
    dt.fit(X, y)
    
    assert dt.tree_ is not None
    assert dt.is_fitted_ == True


def test_unfitted_model_errors():
    """Test that unfitted model raises appropriate errors"""
    dt = DecisionTreeRegressor()
    X = np.array([[1, 1], [2, 2]])
    y = np.array([10, 20])
    
    # All methods should raise errors before fitting
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        dt.predict(X)
    
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        dt.score(X, y)


def test_impurity_functions():
    """Test impurity calculation functions"""
    # Test with constant values (should have 0 variance)
    constant_y = np.array([5, 5, 5, 5])
    assert DecisionTreeRegressor.compute_mse(constant_y) == 0.0
    
    # Test with varying values
    varying_y = np.array([1, 2, 3, 4])
    mse_varying = DecisionTreeRegressor.compute_mse(varying_y)
    mae_varying = DecisionTreeRegressor.compute_mae(varying_y)
    
    assert mse_varying > 0
    assert mae_varying > 0
    
    # Test with empty array
    assert DecisionTreeRegressor.compute_mse(np.array([])) == 0.0
    assert DecisionTreeRegressor.compute_mae(np.array([])) == 0.0

def test_predict_consistency():
    """Test prediction consistency"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X, y)
    
    # Test on training data
    predictions = dt.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    # Test on new points
    new_points = np.array([[1.5, 1.5], [8.5, 8.5]])
    new_predictions = dt.predict(new_points)
    assert len(new_predictions) == 2
    assert all(isinstance(pred, (int, float)) for pred in new_predictions)

def test_different_criteria():
    """Test different splitting criteria"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    # Test MSE criterion
    dt_mse = DecisionTreeRegressor(max_depth=2, criterion='mse')
    dt_mse.fit(X, y)
    predictions_mse = dt_mse.predict(X)
    
    # Test MAE criterion
    dt_mae = DecisionTreeRegressor(max_depth=2, criterion='mae')
    dt_mae.fit(X, y)
    predictions_mae = dt_mae.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_mse) == len(X)
    assert len(predictions_mae) == len(X)

def test_different_max_depths():
    """Test different max depth values"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    max_depths = [1, 2, 3, None]
    for max_depth in max_depths:
        dt = DecisionTreeRegressor(max_depth=max_depth)
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert len(predictions) == len(X)
        assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_score_method():
    """Test score method (R²)"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    dt = DecisionTreeRegressor(max_depth=3)
    dt.fit(X, y)
    
    r2_score = dt.score(X, y)
    
    # R² score should be between 0 and 1 for good fits
    assert 0 <= r2_score <= 1
    
    # For this well-separated data, R² should be high
    assert r2_score > 0.8

def test_fit_predict():
    """Test fit_predict method"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    dt = DecisionTreeRegressor(max_depth=2)
    predictions = dt.fit_predict(X, y)
    
    # Check that model is fitted
    assert dt.is_fitted_ == True
    
    # Check predictions
    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_tree_properties():
    """Test tree depth and leaf count"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    # Test with limited depth
    dt = DecisionTreeRegressor(max_depth=2)
    dt.fit(X, y)
    
    depth = dt.get_depth()
    leaves = dt.get_n_leaves()
    
    assert depth <= 2
    assert leaves >= 1
    assert isinstance(depth, int)
    assert isinstance(leaves, int)

def test_min_samples_constraints():
    """Test minimum samples constraints"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],  # Lower values
                  [8, 8], [8, 9], [9, 8], [9, 9]])  # Higher values
    y = np.array([10, 12, 14, 16, 80, 85, 90, 95], dtype=float)
    
    # Test with high min_samples_split
    dt = DecisionTreeRegressor(min_samples_split=5)
    dt.fit(X, y)
    
    # Should still work but might create simpler tree
    predictions = dt.predict(X)
    assert len(predictions) == len(X)
    
    # Test with high min_samples_leaf
    dt_leaf = DecisionTreeRegressor(min_samples_leaf=3)
    dt_leaf.fit(X, y)
    
    predictions_leaf = dt_leaf.predict(X)
    assert len(predictions_leaf) == len(X)

def test_leaf_values():
    """Test that leaf values are appropriate"""
    X = np.array([[1, 1], [2, 2]])
    y = np.array([10, 20])
    
    # Test MSE criterion (should use mean)
    dt_mse = DecisionTreeRegressor(criterion='mse')
    dt_mse.fit(X, y)
    predictions_mse = dt_mse.predict(X)
    
    # Test MAE criterion (should use median)
    dt_mae = DecisionTreeRegressor(criterion='mae')
    dt_mae.fit(X, y)
    predictions_mae = dt_mae.predict(X)
    
    # Both should produce valid predictions
    assert len(predictions_mse) == 2
    assert len(predictions_mae) == 2

def test_parameter_management():
    """Test parameter get/set methods"""
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=3, verbose=False)
    
    # Test get_params
    params = dt.get_params()
    assert 'max_depth' in params
    assert 'min_samples_split' in params
    assert params['max_depth'] == 5
    assert params['min_samples_split'] == 3
    
    # Test set_params
    result = dt.set_params(max_depth=10, min_samples_split=2)
    assert result is dt  # Should return self
    assert dt.max_depth == 10
    assert dt.min_samples_split == 2

def test_edge_cases():
    """Test edge cases"""
    # Test with single sample
    X_single = np.array([[1, 1]])
    y_single = np.array([10])
    dt_single = DecisionTreeRegressor()
    dt_single.fit(X_single, y_single)
    predictions_single = dt_single.predict(X_single)
    
    assert predictions_single.shape == (1,)
    assert predictions_single[0] == 10  # Should predict the only target value
    
    # Test with identical samples
    X_identical = np.array([[1, 1], [1, 1], [1, 1]])
    y_identical = np.array([10, 10, 10])
    dt_identical = DecisionTreeRegressor()
    dt_identical.fit(X_identical, y_identical)
    predictions_identical = dt_identical.predict(X_identical)
    
    assert len(predictions_identical) == 3
    assert all(pred == 10 for pred in predictions_identical)
    
    # Test with very small dataset
    X_small = np.array([[1], [2]])
    y_small = np.array([10, 20])
    dt_small = DecisionTreeRegressor()
    dt_small.fit(X_small, y_small)
    predictions_small = dt_small.predict(X_small)
    
    assert len(predictions_small) == 2

def test_regression_accuracy():
    """Test regression accuracy on simple linear relationship"""
    # Create simple linear relationship
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # y = 2x
    
    dt = DecisionTreeRegressor(max_depth=None)
    dt.fit(X, y)
    
    # Test predictions
    predictions = dt.predict(X)
    
    # For perfect linear relationship, tree should fit well
    assert len(predictions) == len(X)
    
    # Test R² score
    r2 = dt.score(X, y)
    assert r2 >= 0  # Should be non-negative

def test_information_gain():
    """Test information gain calculation"""
    # Test with clear separation
    y_parent = np.array([1, 1, 1, 10, 10, 10])
    y_left = np.array([1, 1, 1])
    y_right = np.array([10, 10, 10])
    
    dt = DecisionTreeRegressor()
    information_gain = dt._information_gain(y_parent, y_left, y_right)
    
    # Should have positive information gain
    assert information_gain > 0
    
    # Test with no separation
    y_no_split = np.array([5, 5, 5, 5, 5, 5])
    information_gain_no = dt._information_gain(y_no_split, y_no_split, y_no_split)
    
    # Should have zero information gain
    assert information_gain_no == 0.0
