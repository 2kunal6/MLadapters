
import torch.nn.LSTM as LSTM
from MLalgorithms.NeuralNetwork.RNNBase import RNNBase


class LSTM(RNNBase):
	
	def __init__(self, num_layers, bidirectional, batch_first, bias, input_size, hidden_size, dropout):
		RNNBase.__init__(self, num_layers, bidirectional, batch_first, bias, input_size, hidden_size, dropout)
		self.model = LSTM(bidirectional = self.bidirectional,
			num_layers = self.num_layers,
			dropout = self.dropout,
			hidden_size = self.hidden_size,
			bias = self.bias,
			input_size = self.input_size,
			batch_first = self.batch_first)

