import numpy as np

"""
add intercept
loss
gradient

fit - iterate over epochs, predict, gradient, weights, loss
l1
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
     derivative of J(β) with respect to β is 
     ∇J(β) = (1/n) * ∇(yᵀy - 2βᵀXᵀy + βᵀXᵀXβ)

     ∇J(β) = (1/n) * (0 - 2Xᵀy + 2XᵀXβ)
       = (1/n) * 2Xᵀ(Xβ - y)
       = (2/n) * Xᵀ(Xβ - y)


   = (2/n) * Xᵀ(ŷ - y)
"""

class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, verbose: bool = False, random_state: int | None = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.weights: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)

    def _initialize_weights(self, X: np.ndarray) -> None:
        n_features = X.shape[1]
        np.random.seed(self.random_state)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=(n_features,))

    @staticmethod
    def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Mean Squared Error: (1/n) * sum((y - y_hat)^2)
        diff = y_true - y_pred
        return float(np.mean(diff**2))

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
            self._initialize_weights(X_aug)

        for epoch in range(self.epochs):
            y_pred = X_aug @ self.weights
            grad = self._mse_gradient(X_aug, y_vec, y_pred)
            self.weights -= self.learning_rate * grad

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                loss = self._mse_loss(y_vec, y_pred)
                print(f"epoch={epoch:5d} loss={loss:.6f}")


        """ STOCHASTIC GRADIENT DESCENT
        for epoch in range(self.epochs):
            for i in range(X_aug.shape[0]):  # Loop over each sample
                y_pred_i = X_aug[i] @ self.weights  # Single sample
                grad_i = 2.0 * X_aug[i] * (y_pred_i - y_vec[i])  # Single sample gradient
                self.weights -= self.learning_rate * grad_i  # Update per sample
        """
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

    model = LinearRegressionGD(learning_rate=0.05, epochs=2000, verbose=False, random_state=0)
    model.fit(X, y)

    print("Weights (including intercept):", model.weights)
    print("R^2:", model.score(X, y))


# Modified Linear Regression with L1 and L2 Regularization
class LinearRegressionRegularized:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 verbose: bool = False, 
                 random_state: int | None = None,
                 alpha_l1: float = 0.0,  # L1 regularization strength
                 alpha_l2: float = 0.0): # L2 regularization strength
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.weights: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)

    def _initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(n_features,))

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Combined loss: MSE + L1 + L2 regularization"""
        # MSE loss
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Regularization terms (skip intercept)
        if self.weights is None or len(self.weights) <= 1:
            return mse
        feature_weights = self.weights[1:]
        l1_penalty = self.alpha_l1 * np.sum(np.abs(feature_weights))
        l2_penalty = self.alpha_l2 * np.sum(feature_weights ** 2)
        
        return mse + l1_penalty + l2_penalty

    def _gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Combined gradient: MSE + L1 + L2 regularization"""
        # MSE gradient: (2/n) * X^T(ŷ - y)
        n_samples = y_true.shape[0]
        mse_grad = (2.0 / n_samples) * (X.T @ (y_pred - y_true))
        
        # Regularization gradients (skip intercept)
        if self.weights is None or len(self.weights) <= 1:
            return mse_grad
        
        grad = np.zeros_like(self.weights)
        grad[:] = mse_grad
        grad[1:] += self.alpha_l1 * np.sign(self.weights[1:])  # L1 gradient
        grad[1:] += 2.0 * self.alpha_l2 * self.weights[1:]      # L2 gradient
        
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionRegularized":
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        y_vec = np.asarray(y, dtype=float).ravel()

        if self.weights is None:
            self._initialize_weights(X_aug.shape[1])

        for epoch in range(self.epochs):
            y_pred = X_aug @ self.weights
            grad = self._gradient(X_aug, y_vec, y_pred)
            self.weights -= self.learning_rate * grad

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                loss = self._loss(y_vec, y_pred)
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


