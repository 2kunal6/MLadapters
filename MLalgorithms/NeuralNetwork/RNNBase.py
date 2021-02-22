
from MLalgorithms.NeuralNetwork import NeuralNetwork


class RNNBase(NeuralNetwork):
	
	def __init__(self, num_layers = 1, bidirectional = False, batch_first = False, bias = True, input_size, hidden_size, dropout = 0):
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.batch_first = batch_first
		self.bias = bias
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout

