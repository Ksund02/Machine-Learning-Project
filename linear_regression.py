import numpy as np

class LinearRegression():
    
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.weights = None
        self.bias = None
        self.epochs = 100
        self.lr = 0.01
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # Initialize weights and bias
        if (X[0] is list):
            self.weights = np.zeros(X.shape[1])
        else:
            self.weights = 0
        self.bias = 0

        for _ in range(self.epochs):
            #lin_model = np.matmul(self.weights, X.transpose()) + self.bias
            lin_model = self.weights * X + self.bias
            grad_w, grad_b = self.compute_gradients(X, y, lin_model)
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
        
        # lin_model = np.matmul(X, self.weights) + self.bias
        lin_model = self.weights * X + self.bias

        return lin_model

    def compute_gradients(self, X, y, lin_model):
        w_derivative = -2/len(X) * np.sum((y - lin_model) * X)
        b_derivative = -2/len(X) * np.sum(y - lin_model)
        return w_derivative, b_derivative
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.lr * grad_w
        self.weights -= self.lr * grad_b