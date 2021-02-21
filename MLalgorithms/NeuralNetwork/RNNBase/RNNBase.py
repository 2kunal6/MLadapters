from MLalgorithms.NeuralNetwork import NeuralNetwork






class RNNBase(NeuralNetwork):
    
    def __init__(self, dropout = 0, bidirectional = False, hidden_size, bias = True, batch_first = False, input_size, num_layers = 1):
        self.dropout = dropout
		self.bidirectional = bidirectional
		self.hidden_size = hidden_size
		self.bias = bias
		self.batch_first = batch_first
		self.input_size = input_size
		self.num_layers = num_layers

    
