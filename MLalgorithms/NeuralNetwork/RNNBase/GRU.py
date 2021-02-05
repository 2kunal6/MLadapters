


import torch.nn.GRU as GRU


class GRU(RNNBase):
    
    def __init__(self, dropout, batch_first, hidden_size, num_layers, input_size, bidirectional, bias):
        RNNBase.__init__(self, dropout, batch_first, hidden_size, num_layers, input_size, bidirectional, bias)
		self.model = GRU(bidirectional = self.bidirectional,
			bias = self.bias,
			hidden_size = self.hidden_size,
			dropout = self.dropout,
			batch_first = self.batch_first,
			num_layers = self.num_layers,
			input_size = self.input_size)
    
