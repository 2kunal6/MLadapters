
import torch.nn.RNN as RNN
from MLalgorithms.NeuralNetwork.RNNBase import RNNBase


class RNN(RNNBase):
	
	def __init__(self, bidirectional, input_size, bias, batch_first, hidden_size, num_layers, dropout, nonlinearity = 'tanh'):
		self.nonlinearity = nonlinearity
		RNNBase.__init__(self, bidirectional, input_size, bias, batch_first, hidden_size, num_layers, dropout)
		self.model = RNN(bias = self.bias,
			batch_first = self.batch_first,
			nonlinearity = self.nonlinearity,
			hidden_size = self.hidden_size,
			dropout = self.dropout,
			bidirectional = self.bidirectional,
			num_layers = self.num_layers,
			input_size = self.input_size)

