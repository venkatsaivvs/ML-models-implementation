import numpy as np

"""
Linear Regression Mathematical Formulation:

1. Prediction: ŷ = Xβ
   where X is the feature matrix (n × p), β is the weight vector (p × 1)

2. Loss Function (MSE): J(β) = (1/n) * Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
   = (1/n) * (y - Xβ)ᵀ(y - Xβ)
   = (1/n) * ||y - Xβ||²




3. Gradient: ∇J(β) = (2/n) * Xᵀ(Xβ - y)
J(β) = (1/n) * (y - Xβ)ᵀ(y - Xβ)
     = (1/n) * (yᵀ - βᵀXᵀ)(y - Xβ)
     = (1/n) * (yᵀy - yᵀXβ - βᵀXᵀy + βᵀXᵀXβ)
     J(β) = (1/n) * (yᵀy - 2βᵀXᵀy + βᵀXᵀXβ)
     ∇J(β) = (1/n) * ∇(yᵀy - 2βᵀXᵀy + βᵀXᵀXβ)

     ∇J(β) = (1/n) * (0 - 2Xᵀy + 2XᵀXβ)
       = (1/n) * 2Xᵀ(Xβ - y)
       = (2/n) * Xᵀ(Xβ - y)


   = (2/n) * Xᵀ(ŷ - y)
"""

class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, fit_intercept: bool = True, verbose: bool = False, random_state: int | None = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state
        self.weights: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)

    def _initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(n_features,))

    @staticmethod
    def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Mean Squared Error: (1/n) * sum((y - y_hat)^2)
        n_samples = y_true.shape[0]
        diff = y_pred - y_true
        return float((diff @ diff) / n_samples)

    @staticmethod
    def _mse_gradient(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Gradient of MSE w.r.t. weights: ∇J(β) = (2/n) * X^T(ŷ - y)
        n_samples = y_true.shape[0]
        return (2.0 / n_samples) * (X.T @ (y_pred - y_true))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
            assert y.ndim == 1, f"y should be 1D after ravel, got shape {y.shape}"
            assert len(y) > 0, "y should not be empty after ravel"

        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        y_vec = np.asarray(y, dtype=float).ravel()

        if self.weights is None:
            self._initialize_weights(X_aug.shape[1])

        for epoch in range(self.epochs):
            y_pred = X_aug @ self.weights
            grad = self._mse_gradient(X_aug, y_vec, y_pred)
            self.weights -= self.learning_rate * grad

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                loss = self._mse_loss(y_vec, y_pred)
                print(f"epoch={epoch:5d} loss={loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        return X_aug @ self.weights

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # R^2 score
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # Guard against division by zero when y is constant
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_samples = 300
    X = rng.normal(size=(n_samples, 2))
    true_intercept = 1.5
    true_weights = np.array([2.0, -3.0])
    noise = rng.normal(scale=0.5, size=n_samples)
    y = true_intercept + X @ true_weights + noise

    model = LinearRegressionGD(learning_rate=0.05, epochs=2000, fit_intercept=True, verbose=False, random_state=0)
    model.fit(X, y)

    print("Weights (including intercept if enabled):", model.weights)
    print("R^2:", model.score(X, y))


# Modified Linear Regression with L1 and L2 Regularization
class LinearRegressionRegularized:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 fit_intercept: bool = True, verbose: bool = False, 
                 random_state: int | None = None,
                 alpha_l1: float = 0.0,  # L1 regularization strength
                 alpha_l2: float = 0.0): # L2 regularization strength
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.weights: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)

    def _initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(n_features,))

    @staticmethod
    def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Mean Squared Error: (1/n) * sum((y - y_hat)^2)
        n_samples = y_true.shape[0]
        diff = y_pred - y_true
        return float((diff @ diff) / n_samples)

    def _l1_penalty(self) -> float:
        """Calculate L1 regularization penalty"""
        if self.weights is None:
            return 0.0
        # Don't regularize intercept if fit_intercept is True
        weights_to_regularize = self.weights[1:] if self.fit_intercept else self.weights
        return self.alpha_l1 * np.sum(np.abs(weights_to_regularize))

    def _l2_penalty(self) -> float:
        """Calculate L2 regularization penalty"""
        if self.weights is None:
            return 0.0
        # Don't regularize intercept if fit_intercept is True
        weights_to_regularize = self.weights[1:] if self.fit_intercept else self.weights
        return self.alpha_l2 * np.sum(weights_to_regularize ** 2)

    def _regularized_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate total loss including regularization"""
        mse_loss = self._mse_loss(y_true, y_pred)
        l1_penalty = self._l1_penalty()
        l2_penalty = self._l2_penalty()
        return mse_loss + l1_penalty + l2_penalty

    def _mse_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Base MSE gradient: (2/n) * X^T(ŷ - y)"""
        n_samples = y_true.shape[0]
        return (2.0 / n_samples) * (X.T @ (y_pred - y_true))

    def _l1_gradient(self) -> np.ndarray:
        """L1 regularization gradient: α₁ * sign(β)"""
        if self.weights is None:
            return np.zeros(0)
        
        # Create gradient array same shape as weights
        l1_grad = np.zeros_like(self.weights)
        
        # Don't regularize intercept if fit_intercept is True
        if self.fit_intercept and len(self.weights) > 1:
            # Regularize all weights except intercept (first element)
            feature_weights = self.weights[1:]
            l1_grad[1:] = self.alpha_l1 * np.sign(feature_weights)
        else:
            # Regularize all weights
            l1_grad = self.alpha_l1 * np.sign(self.weights)
        
        return l1_grad

    def _l2_gradient(self) -> np.ndarray:
        """L2 regularization gradient: 2α₂β"""
        if self.weights is None:
            return np.zeros(0)
        
        # Create gradient array same shape as weights
        l2_grad = np.zeros_like(self.weights)
        
        # Don't regularize intercept if fit_intercept is True
        if self.fit_intercept and len(self.weights) > 1:
            # Regularize all weights except intercept (first element)
            l2_grad[1:] = 2.0 * self.alpha_l2 * self.weights[1:]
        else:
            # Regularize all weights
            l2_grad = 2.0 * self.alpha_l2 * self.weights
        
        return l2_grad

    def _regularized_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Combined gradient with L1 and L2 regularization"""
        mse_grad = self._mse_gradient(X, y_true, y_pred)
        l1_grad = self._l1_gradient()
        l2_grad = self._l2_gradient()
        return mse_grad + l1_grad + l2_grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionRegularized":
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        y_vec = np.asarray(y, dtype=float).ravel()

        if self.weights is None:
            self._initialize_weights(X_aug.shape[1])

        for epoch in range(self.epochs):
            y_pred = X_aug @ self.weights
            grad = self._regularized_gradient(X_aug, y_vec, y_pred)
            self.weights -= self.learning_rate * grad

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                loss = self._regularized_loss(y_vec, y_pred)
                mse_loss = self._mse_loss(y_vec, y_pred)
                l1_penalty = self._l1_penalty()
                l2_penalty = self._l2_penalty()
                print(f"epoch={epoch:5d} total_loss={loss:.6f} mse={mse_loss:.6f} l1={l1_penalty:.6f} l2={l2_penalty:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        return X_aug @ self.weights

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # R^2 score
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # Guard against division by zero when y is constant
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


# Example usage with regularization
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Regularized Linear Regression")
    print("="*50)
    
    rng = np.random.default_rng(42)
    n_samples = 300
    X = rng.normal(size=(n_samples, 2))
    true_intercept = 1.5
    true_weights = np.array([2.0, -3.0])
    noise = rng.normal(scale=0.5, size=n_samples)
    y = true_intercept + X @ true_weights + noise

    # Test different regularization approaches
    models = {
        "No Regularization": LinearRegressionRegularized(alpha_l1=0.0, alpha_l2=0.0, learning_rate=0.05, epochs=2000, verbose=False),
        "L2 Regularization (Ridge)": LinearRegressionRegularized(alpha_l1=0.0, alpha_l2=1.0, learning_rate=0.05, epochs=2000, verbose=False),
        "L1 Regularization (Lasso)": LinearRegressionRegularized(alpha_l1=0.1, alpha_l2=0.0, learning_rate=0.05, epochs=2000, verbose=False),
        "Elastic Net": LinearRegressionRegularized(alpha_l1=0.05, alpha_l2=0.5, learning_rate=0.05, epochs=2000, verbose=False)
    }

    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X, y)
        print(f"  Weights: {model.weights}")
        print(f"  R² Score: {model.score(X, y):.6f}")


