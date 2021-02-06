


import torch.nn.RNN as RNN


class RNN(RNNBase):
    
    def __init__(self, input_size, bidirectional, dropout, num_layers, bias, hidden_size, batch_first, nonlinearity = 'tanh'):
        self.nonlinearity = nonlinearity
		RNNBase.__init__(self, input_size, bidirectional, dropout, num_layers, bias, hidden_size, batch_first)
		self.model = RNN(hidden_size = self.hidden_size,
			batch_first = self.batch_first,
			dropout = self.dropout,
			input_size = self.input_size,
			nonlinearity = self.nonlinearity,
			bias = self.bias,
			bidirectional = self.bidirectional,
			num_layers = self.num_layers)
    
    
