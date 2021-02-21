from MLalgorithms.NeuralNetwork.RNNBase import RNNBase



import torch.nn.LSTM as LSTM


class LSTM(RNNBase):
    
    def __init__(self, dropout, bidirectional, hidden_size, bias, batch_first, input_size, num_layers):
        RNNBase.__init__(self, dropout, bidirectional, hidden_size, bias, batch_first, input_size, num_layers)
		self.model = LSTM(bidirectional = self.bidirectional,
			input_size = self.input_size,
			batch_first = self.batch_first,
			hidden_size = self.hidden_size,
			dropout = self.dropout,
			bias = self.bias,
			num_layers = self.num_layers)
    
