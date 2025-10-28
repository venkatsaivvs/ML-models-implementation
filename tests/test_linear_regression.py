from src.ml_code.linear_regression import LinearRegressionGD
import numpy as np
import pytest
#tdd test driven development approach


def test_linear_regression_gd():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    lr = LinearRegressionGD()
    lr.fit(X, y)
    assert lr.coef_ is not None
    assert lr.intercept_ is not None
    assert lr.predict(X) is not None
    assert lr.score(X, y) is not None
    assert lr.predict(X) is not None
    # Test that weights are properly initialized
    assert lr.weights is not None
    assert len(lr.weights) == 3  # 2 features + 1 intercept
    print(f"Weights: {lr.weights}")  # This will show with -s flag

#write additional tests for the linear regression model - I am following a test driven development approach

def test_empty_data():
    X = np.array([])
    y = np.array([])
    lr = LinearRegressionGD()
    with pytest.raises(AssertionError):
        lr.fit(X, y)


def test_shape_mismatch():
    """Test that model handles shape mismatches between X and y appropriately"""
    X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
    y = np.array([1, 2])  # Only 2 samples - mismatch!
    lr = LinearRegressionGD()
    
    with pytest.raises(ValueError):
        lr.fit(X, y)


def test_unfitted_model_prediction():
    """Test that calling predict() before fit() raises appropriate error"""
    X = np.array([[1, 2], [3, 4]])
    lr = LinearRegressionGD()
    
    # Should raise error when trying to predict without fitting
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        lr.predict(X)
