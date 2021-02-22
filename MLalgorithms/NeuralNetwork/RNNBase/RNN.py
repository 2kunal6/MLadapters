
import torch.nn.RNN as RNN
from MLalgorithms.NeuralNetwork.RNNBase import RNNBase


class RNN(RNNBase):
	
	def __init__(self, num_layers, bidirectional, batch_first, bias, input_size, hidden_size, dropout, nonlinearity = 'tanh'):
		self.nonlinearity = nonlinearity
		RNNBase.__init__(self, num_layers, bidirectional, batch_first, bias, input_size, hidden_size, dropout)
		self.model = RNN(bidirectional = self.bidirectional,
			num_layers = self.num_layers,
			dropout = self.dropout,
			hidden_size = self.hidden_size,
			nonlinearity = self.nonlinearity,
			bias = self.bias,
			input_size = self.input_size,
			batch_first = self.batch_first)

