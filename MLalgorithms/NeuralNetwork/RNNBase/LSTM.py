
import torch.nn.LSTM as LSTM
from MLalgorithms.NeuralNetwork.RNNBase import RNNBase


class LSTM(RNNBase):
	
	def __init__(self, bidirectional, input_size, bias, batch_first, hidden_size, num_layers, dropout):
		RNNBase.__init__(self, bidirectional, input_size, bias, batch_first, hidden_size, num_layers, dropout)
		self.model = LSTM(bias = self.bias,
			batch_first = self.batch_first,
			hidden_size = self.hidden_size,
			dropout = self.dropout,
			bidirectional = self.bidirectional,
			num_layers = self.num_layers,
			input_size = self.input_size)

