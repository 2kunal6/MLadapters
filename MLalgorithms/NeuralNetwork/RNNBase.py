





class RNNBase(NeuralNetwork):
    
    def __init__(self, input_size, bidirectional = False, dropout = 0, num_layers = 1, bias = True, hidden_size, batch_first = False):
        self.input_size = input_size
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.num_layers = num_layers
		self.bias = bias
		self.hidden_size = hidden_size
		self.batch_first = batch_first
    
    
