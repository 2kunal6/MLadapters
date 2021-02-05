





class RNNBase(NeuralNetwork):
    
    def __init__(self, dropout = 0, batch_first = False, hidden_size, num_layers = 1, input_size, bidirectional = False, bias = True):
        self.dropout = dropout
		self.batch_first = batch_first
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.input_size = input_size
		self.bidirectional = bidirectional
		self.bias = bias
    
    
