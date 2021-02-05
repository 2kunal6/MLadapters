





class RNNBase(NeuralNetwork):
    
    def __init__(self, bias = True, dropout = 0, bidirectional = False, hidden_size, batch_first = False, num_layers = 1, input_size):
        self.bias = bias
		self.dropout = dropout
		self.bidirectional = bidirectional
		self.hidden_size = hidden_size
		self.batch_first = batch_first
		self.num_layers = num_layers
		self.input_size = input_size

    
