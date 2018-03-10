import Layer
# This module should implement a network using multiple layers from layer.py
# Future enhancment, to implement hyper-parameter tuning mechanism to find an optimal network setup 
class Network():
    """Neural Network implementation"""
    def __init__(self, X, y, layers):
        """Initialize a network given at input and output shapes and network structure of hidden and output layers"""
        # check shape consistency 
        n_input, _m_i = X.shape
        n_output, _m_o = y.shape
        if _m_i == _m_o:
            # Input layer
            self.layers[0] = Layer(n_input, 
                                   layers["n1"], 'input', 0, 0, 0, 0, False)
            # Hidden layers
            for layer in range(1,layers["num_layers"]-1):
                self.layers[layer] = Layer(layers["n"+str(layer)],                                        
                                           layers["n"+str(layer+1)],                                        
                                           layers["act"+str(layer)],                                        
                                           layers["alpha"+str(layer)],                                        
                                           layers["lambda"+str(layer)],                                        
                                           layers["p"+str(layer)],                                        
                                           layers["gamma"+str(layer)],                                        
                                           False)
             # Output layer 
            layer += 1
            self.layers[layer] = Layer(layers["n"+str(layer)],                                           
                                       n_output,
                                       layers["act"+str(layer)],
                                       layers["alpha"+str(layer)],
                                       layers["lambda"+str(layer)],
                                       layers["p"+str(layer)],
                                       layers["gamma"+str(layer)],
                                       True)
#TODO: add a Re-Inforcement Learning agent to tune the network, sending signals to individual layers to optimize hyperparameters
