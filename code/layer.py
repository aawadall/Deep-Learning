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
    def __init__(self, n_prev, n, act='sigmoid', alpha=0.5, _lambda=0.1, p =0.75):
        """Neural Network initialization method, 
        given previous layer dimensions n_prev, 
        and current layer dimensions n, 
        construct weight matrix W and intercept (bias vector) b"""
        
