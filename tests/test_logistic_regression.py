from src.ml_code.logistic_regression import LogisticRegressionGD
import numpy as np
from pytest import raises
import pytest

#tdd test driven development approach

def test_logistic_regression_gd():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    lr = LogisticRegressionGD()
    lr.fit(X, y)
    assert lr.coef_ is not None
    assert lr.intercept_ is not None
    assert lr.predict(X) is not None
    assert lr.score(X, y) is not None


def test_empty_data():
    X = np.array([])
    y = np.array([])
    lr = LogisticRegressionGD()
    with pytest.raises(AssertionError):
        lr.fit(X, y)


def test_shape_mismatch():
    """Test that model handles shape mismatches between X and y appropriately"""
    X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
    y = np.array([1, 2])  # Only 2 samples - mismatch!
    lr = LogisticRegressionGD()
    
    with pytest.raises(ValueError):
        lr.fit(X, y)


def test_unfitted_model_prediction():
    """Test that calling predict() before fit() raises appropriate error"""
    X = np.array([[1, 2], [3, 4]])
    lr = LogisticRegressionGD()
    
    # Should raise error when trying to predict without fitting
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        lr.predict(X)
