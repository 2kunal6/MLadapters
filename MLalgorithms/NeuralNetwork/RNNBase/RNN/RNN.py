from MLalgorithms.NeuralNetwork.RNNBase import RNNBase



import torch.nn.RNN as RNN


class RNN(RNNBase):
    
    def __init__(self, dropout, bidirectional, hidden_size, bias, batch_first, input_size, num_layers, nonlinearity = 'tanh'):
        self.nonlinearity = nonlinearity
		RNNBase.__init__(self, dropout, bidirectional, hidden_size, bias, batch_first, input_size, num_layers)
		self.model = RNN(bidirectional = self.bidirectional,
			input_size = self.input_size,
			batch_first = self.batch_first,
			bias = self.bias,
			dropout = self.dropout,
			hidden_size = self.hidden_size,
			nonlinearity = self.nonlinearity,
			num_layers = self.num_layers)

    
