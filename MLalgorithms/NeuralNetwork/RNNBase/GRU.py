from MLalgorithms.NeuralNetwork.RNNBase import RNNBase



import torch.nn.GRU as GRU


class GRU(RNNBase):
    
    def __init__(self, hidden_size, bias, batch_first, input_size, num_layers, dropout, bidirectional):
        RNNBase.__init__(self, hidden_size, bias, batch_first, input_size, num_layers, dropout, bidirectional)
		self.model = GRU(bidirectional = self.bidirectional,
			hidden_size = self.hidden_size,
			bias = self.bias,
			num_layers = self.num_layers,
			batch_first = self.batch_first,
			dropout = self.dropout,
			input_size = self.input_size)
    
