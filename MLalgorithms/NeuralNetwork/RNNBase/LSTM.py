


import torch.nn.LSTM as LSTM


class LSTM(RNNBase):
    
    def __init__(self, bias, dropout, bidirectional, hidden_size, batch_first, num_layers, input_size):
        RNNBase.__init__(self, bias, dropout, bidirectional, hidden_size, batch_first, num_layers, input_size)
		self.model = LSTM(num_layers = self.num_layers,
			dropout = self.dropout,
			bidirectional = self.bidirectional,
			input_size = self.input_size,
			bias = self.bias,
			hidden_size = self.hidden_size,
			batch_first = self.batch_first)
    
