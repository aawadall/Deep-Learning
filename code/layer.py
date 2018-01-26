import numpy as np
# Each layer in this network is assumed to have consistent activation functions
# A layer should have the following
# Weights matrix W
# Intercept (bias) vector b
# Activation function act
# learning rate alpha 
# L2 Norm regularization parameter _lambda
# dropout regularization parameter p (as in keeping probability)


class Layer:
    """Neural Network Layer"""
    def __init__(self, n_prev, n, act='sigmoid', alpha=0.5, _lambda=0.1, p=0.75, gamma=0.01, is_output=False):
        """Neural Network initialization method, 
        given previous layer dimensions n_prev, 
        and current layer dimensions n, 
        construct weight matrix W with random numbers multiplied by scaling factor gamma
        and intercept (bias vector) b"""
        self.W = np.random.randn(n, n_prev) * gamma
        self.b = np.zeros((n, 1))
        self.act = act
        self.alpha = alpha
        self._lambda = _lambda
        self.p = p
        self.cached_input = np.zeros(n_prev, 1)
        self._cached_z = np.zeros(n, 1)
        self._cached_a = np.zeros(n, 1)
        self.output_layer = is_output

    def activation(self, z):
        """current layer activation function (g)"""
        if self.act == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.act == 'tanh':
            return np.tanh(z)
        elif self.act == 'LReLU':
            return ((z > 0) * z) + ((z <= 0) * z / 100)
        else:
            return z

    def activation_derivative(self, z):
        """derivative of current layer activation function (g')"""
        if self.act == 'sigmoid':
            return self.activation(z) * (1 - self.activation(z))
        elif self.act == 'tanh':
            return 1-np.power(z, 2)
        elif self.act == 'LReLU':
            return ((z > 0) * 1) + ((z <= 0) * 1 / 100)
        else:
            return z

    def cost(self, a, expected):
        """cost function of the current layer"""
        _y_hat = self.activation(np.dot(self.W, a) + self.b)
        _m = expected.shape[1]
        _log_probs = np.multiply(np.log(_y_hat), expected) + np.multiply(np.log(1 - _y_hat), 1 - expected)
        return - np.sum(_log_probs) / _m
    
    def forward_propagate(self, a_prev):
        """given output of previous layer, and knowing layer parameters, calculate outcome of this later"""
        self._cached_z = np.dot(self.W,a_prev) + self.b
        self._cached_a = self.activation(self._cached_z)
        return self._cached_a
    
    def backward_propagation(self, expected):
        """given an expected outcome, update weights"""
        _m = expected.shape[1]
        
        # If this layer was an output layer, then we use only difference between expected and actual outcome
        if self.output_layer:
            dZ = np.reshape(self._cached_a - expected,(1,-1))
        else:
            dZ = expected * self.activation_derivative(self._cached_a)
        # Calculate Weights gradients
        dW = 1/_m * np.dot(dZ, self.chached_input.T) + self._lambda/_m * self.W
        db = 1/_m * np.sum(dZ, axis=1, keepdims=True)
        # Update parameters 
        self.W *= -self.alpha * dW
        self.b *= -self.alpha * db
