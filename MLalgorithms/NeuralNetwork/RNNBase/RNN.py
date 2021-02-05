


import torch.nn.RNN as RNN


class RNN(RNNBase):
    
    def __init__(self, bias, dropout, bidirectional, hidden_size, batch_first, num_layers, input_size, nonlinearity = 'tanh'):
        self.nonlinearity = nonlinearity
		RNNBase.__init__(self, bias, dropout, bidirectional, hidden_size, batch_first, num_layers, input_size)
		self.model = RNN(num_layers = self.num_layers,
			dropout = self.dropout,
			bidirectional = self.bidirectional,
			input_size = self.input_size,
			bias = self.bias,
			hidden_size = self.hidden_size,
			nonlinearity = self.nonlinearity,
			batch_first = self.batch_first)

    
