
import torch.nn as nn
from functools import reduce
from MLalgorithms._MLalgorithms import MLalgorithms


class NeuralNetwork(nn.Module, MLalgorithms):
	
	def forward(self, X):
		return reduce(lambda X, l: l(X), self.layers, X)

	def __init__(self, layers):
		super(NeuralNetwork, self).__init__()
		self.layers = nn.Sequential(layers)

