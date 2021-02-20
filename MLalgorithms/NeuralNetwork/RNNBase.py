from MLalgorithms.NeuralNetwork import NeuralNetwork






class RNNBase(NeuralNetwork):
    
    def __init__(self, hidden_size, bias = True, batch_first = False, input_size, num_layers = 1, dropout = 0, bidirectional = False):
        self.hidden_size = hidden_size
		self.bias = bias
		self.batch_first = batch_first
		self.input_size = input_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.bidirectional = bidirectional

    
