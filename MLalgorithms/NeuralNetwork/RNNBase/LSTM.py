


import torch.nn.LSTM as LSTM


class LSTM(RNNBase):
    
    def __init__(self, dropout, batch_first, hidden_size, num_layers, input_size, bidirectional, bias):
        RNNBase.__init__(self, dropout, batch_first, hidden_size, num_layers, input_size, bidirectional, bias)
		self.model = LSTM(bidirectional = self.bidirectional,
			bias = self.bias,
			hidden_size = self.hidden_size,
			dropout = self.dropout,
			batch_first = self.batch_first,
			num_layers = self.num_layers,
			input_size = self.input_size)
    
