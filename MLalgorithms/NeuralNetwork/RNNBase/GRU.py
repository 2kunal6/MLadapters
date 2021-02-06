


import torch.nn.GRU as GRU


class GRU(RNNBase):
    
    def __init__(self, input_size, bidirectional, dropout, num_layers, bias, hidden_size, batch_first):
        RNNBase.__init__(self, input_size, bidirectional, dropout, num_layers, bias, hidden_size, batch_first)
		self.model = GRU(hidden_size = self.hidden_size,
			batch_first = self.batch_first,
			dropout = self.dropout,
			input_size = self.input_size,
			bias = self.bias,
			bidirectional = self.bidirectional,
			num_layers = self.num_layers)
    
