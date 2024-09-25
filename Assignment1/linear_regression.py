import numpy as np

class LinearRegression():
    
    def __init__(self, lr=0.001, epochs=1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.weights = None
        self.bias = 0
        self.epochs = epochs
        self.lr = lr
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        if len(X.shape) == 1:
            X = np.array(X).reshape(-1, 1)
        else:
            X = np.array(X)

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])

        # Gradient Descent
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            grad_w, grad_b = self.compute_gradients(X, y, len(X), y_pred)
            self.update_parameters(grad_w, grad_b)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        if len(X.shape) == 1:
            X = np.array(X).reshape(-1, 1)
        else:
            X = np.array(X)
        return np.matmul(X, self.weights) + self.bias

    def compute_gradients(self, X, y, n, y_pred):
        grad_w = -(2 / n) * np.matmul(X.T, (y - y_pred))
        grad_b = -(2 / n) * np.sum(y - y_pred)
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b