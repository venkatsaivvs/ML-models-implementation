from src.ml_code.logistic_regression import LogisticRegressionGD
import numpy as np

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