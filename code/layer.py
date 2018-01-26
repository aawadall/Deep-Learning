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
    def __init__(self, n_prev, n, act='sigmoid', alpha=0.5, _lambda=0.1, p=0.75, gamma=0.01):
        """Neural Network initialization method, 
        given previous layer dimensions n_prev, 
        and current layer dimensions n, 
        construct weight matrix W with random numbers multiplied by scaling factor gamma
        and intercept (bias vector) b"""
        self.W = np.random.randn(n, n_prev) * gamma
        self.b = np.zeros((n, 1))
        self.act = act
        self.alpha = alpha
        self.p = p

    def activation(self, Z):
        if self.act == 'sigmoid':
            return 1 / (1 +np.exp(-Z))
        elif self.act == 'tanh':
            return np.tanh(Z)
        elif self.act == 'LReLU':
            return ((Z > 0) * Z) + ((Z <= 0) * Z / 100)
        else:
            return Z