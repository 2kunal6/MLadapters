


import torch.nn.RNN as RNN


class RNN(RNNBase):
    
    def __init__(self, dropout, batch_first, hidden_size, num_layers, input_size, bidirectional, bias, nonlinearity = 'tanh'):
        self.nonlinearity = nonlinearity
		RNNBase.__init__(self, dropout, batch_first, hidden_size, num_layers, input_size, bidirectional, bias)
		self.model = RNN(bidirectional = self.bidirectional,
			bias = self.bias,
			hidden_size = self.hidden_size,
			nonlinearity = self.nonlinearity,
			dropout = self.dropout,
			batch_first = self.batch_first,
			num_layers = self.num_layers,
			input_size = self.input_size)
    
    
