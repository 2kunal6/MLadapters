
import torch.nn.GRU as GRU
from MLalgorithms.NeuralNetwork.RNNBase import RNNBase


class GRU(RNNBase):
	
	def __init__(self, bidirectional, input_size, bias, batch_first, hidden_size, num_layers, dropout):
		RNNBase.__init__(self, bidirectional, input_size, bias, batch_first, hidden_size, num_layers, dropout)
		self.model = GRU(bias = self.bias,
			batch_first = self.batch_first,
			hidden_size = self.hidden_size,
			dropout = self.dropout,
			bidirectional = self.bidirectional,
			num_layers = self.num_layers,
			input_size = self.input_size)

