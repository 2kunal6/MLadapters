
import torch.nn.GRU as GRU
from MLalgorithms.NeuralNetwork.RNNBase import RNNBase


class GRU(RNNBase):
	
	def __init__(self, num_layers, bidirectional, batch_first, bias, input_size, hidden_size, dropout):
		RNNBase.__init__(self, num_layers, bidirectional, batch_first, bias, input_size, hidden_size, dropout)
		self.model = GRU(bidirectional = self.bidirectional,
			num_layers = self.num_layers,
			dropout = self.dropout,
			hidden_size = self.hidden_size,
			bias = self.bias,
			input_size = self.input_size,
			batch_first = self.batch_first)

