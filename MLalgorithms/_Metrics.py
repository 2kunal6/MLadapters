
from MLalgorithms._MLalgorithms import MLalgorithms


class Metrics(MLalgorithms):
	
	def __init__(self, sample_weight = None, labels = None, y_true):
		self.sample_weight = sample_weight
		self.labels = labels
		self.y_true = y_true

