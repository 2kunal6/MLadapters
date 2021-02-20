from MLalgorithms.NeuralNetwork.RNNBase import RNNBase



import torch.nn.RNN as RNN


class RNN(RNNBase):
    
    def __init__(self, hidden_size, bias, batch_first, input_size, num_layers, dropout, bidirectional, nonlinearity = 'tanh'):
        self.nonlinearity = nonlinearity
		RNNBase.__init__(self, hidden_size, bias, batch_first, input_size, num_layers, dropout, bidirectional)
		self.model = RNN(bidirectional = self.bidirectional,
			hidden_size = self.hidden_size,
			bias = self.bias,
			num_layers = self.num_layers,
			batch_first = self.batch_first,
			dropout = self.dropout,
			nonlinearity = self.nonlinearity,
			input_size = self.input_size)

    
