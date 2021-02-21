
from MLalgorithms.NeuralNetwork import NeuralNetwork


class RNNBase(NeuralNetwork):
	
	def __init__(self, bidirectional = False, input_size, bias = True, batch_first = False, hidden_size, num_layers = 1, dropout = 0):
		self.bidirectional = bidirectional
		self.input_size = input_size
		self.bias = bias
		self.batch_first = batch_first
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout

