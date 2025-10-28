import numpy as np

"""
Logistic Regression Mathematical Formulation:

add intercept
loss - corss entropy oss 
gradient

fit - iterate over epochs, predict, gradient, weights, loss

1. Prediction: ŷ = σ(Xβ) = 1/(1 + e^(-Xβ))
   where σ is the sigmoid function, X is the feature matrix (n × p), β is the weight vector (p × 1)

2. Loss Function (Cross-Entropy): J(β) = -(1/n) * Σᵢ₌₁ⁿ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
   where yᵢ ∈ {0, 1} are binary labels

3. Gradient: ∇J(β) = (1/n) * Xᵀ(ŷ - y)
   Derivation:
   J(β) = -(1/n) * Σᵢ₌₁ⁿ [yᵢ log(σ(Xᵢβ)) + (1-yᵢ) log(1-σ(Xᵢβ))]
   
   Using chain rule and σ'(z) = σ(z)(1-σ(z)):
   ∂J/∂βⱼ = -(1/n) * Σᵢ₌₁ⁿ [yᵢ * (1/σ(Xᵢβ)) * σ(Xᵢβ)(1-σ(Xᵢβ)) * Xᵢⱼ + 
                            (1-yᵢ) * (1/(1-σ(Xᵢβ))) * (-σ(Xᵢβ)(1-σ(Xᵢβ))) * Xᵢⱼ]
   
   Simplifying:
   ∂J/∂βⱼ = -(1/n) * Σᵢ₌₁ⁿ [yᵢ(1-σ(Xᵢβ))Xᵢⱼ - (1-yᵢ)σ(Xᵢβ)Xᵢⱼ]
          = -(1/n) * Σᵢ₌₁ⁿ [yᵢXᵢⱼ - yᵢσ(Xᵢβ)Xᵢⱼ - (1-yᵢ)σ(Xᵢβ)Xᵢⱼ]
          = -(1/n) * Σᵢ₌₁ⁿ [yᵢXᵢⱼ - σ(Xᵢβ)Xᵢⱼ]
          = (1/n) * Σᵢ₌₁ⁿ [σ(Xᵢβ) - yᵢ]Xᵢⱼ
   
   Therefore: ∇J(β) = (1/n) * Xᵀ(ŷ - y)
"""


class LogisticRegressionGD:
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
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Sigmoid function: σ(z) = 1/(1 + e^(-z))
        # Clip z to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Cross-entropy loss: -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        n_samples = y_true.shape[0]
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(np.mean(loss))

    @staticmethod
    def _cross_entropy_gradient(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Gradient of cross-entropy w.r.t. weights: ∇J(β) = (1/n) * X^T(ŷ - y)
        n_samples = y_true.shape[0]
        return (1.0 / n_samples) * (X.T @ (y_pred - y_true))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        y_vec = np.asarray(y, dtype=float).ravel()
        
        # Ensure binary labels
        unique_labels = np.unique(y_vec)
        if len(unique_labels) != 2:
            raise ValueError("Logistic regression requires exactly 2 classes")
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("Labels must be 0 and 1 for binary classification")

        if self.weights is None:
            self._initialize_weights(X_aug.shape[1])

        for epoch in range(self.epochs):
            # Forward pass: compute predictions
            z = X_aug @ self.weights
            y_pred = self._sigmoid(z)
            
            # Backward pass: compute gradient and update weights
            grad = self._cross_entropy_gradient(X_aug, y_vec, y_pred)
            self.weights -= self.learning_rate * grad

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                loss = self._cross_entropy_loss(y_vec, y_pred)
                print(f"epoch={epoch:5d} loss={loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
        X_aug = self._add_intercept(np.asarray(X, dtype=float))
        z = X_aug @ self.weights
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels using threshold"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Accuracy score
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_samples = 300
    
    # Generate synthetic binary classification data
    X = rng.normal(size=(n_samples, 2))
    true_intercept = 0.5
    true_weights = np.array([2.0, -1.5])
    
    # Create linear decision boundary with noise
    z_true = true_intercept + X @ true_weights
    p_true = 1.0 / (1.0 + np.exp(-z_true))  # True probabilities
    y = (rng.random(n_samples) < p_true).astype(int)  # Binary labels
    
    model = LogisticRegressionGD(learning_rate=0.1, epochs=1000, fit_intercept=True, verbose=False, random_state=0)
    model.fit(X, y)
    
    print("Weights (including intercept if enabled):", model.weights)
    print("Accuracy:", model.score(X, y))
    
    # Show some predictions
    print("\nSample predictions:")
    sample_indices = rng.choice(n_samples, 5, replace=False)
    for idx in sample_indices:
        prob = model.predict_proba(X[idx:idx+1])[0]
        pred = model.predict(X[idx:idx+1])[0]
        print(f"Sample {idx}: True={y[idx]}, Pred={pred}, Prob={prob:.3f}")




#3. Multinomial Logistic Regression (Softmax)

# Instead of single weight vector
self.weights: np.ndarray  # shape: (n_features,)

# We'd have weight matrix
self.weights: np.ndarray  # shape: (n_features, n_classes)

# Instead of sigmoid
def _softmax(self, z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Instead of binary cross-entropy
def _multinomial_cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # y_true: (n_samples, n_classes) - one-hot encoded
    # y_pred: (n_samples, n_classes) - softmax probabilities
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred_clipped))
    return float(loss / y_true.shape[0])

# Gradient becomes
def _multinomial_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # ∇J(β) = (1/n) * X^T(ŷ - y)
    return (1.0 / X.shape[0]) * (X.T @ (y_pred - y_true))