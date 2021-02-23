
from sklearn.metrics import confusion_matrix as CM
from MLalgorithms._Metrics import Metrics


class confusion_matrix(Metrics):
	
	def __init__(self, sample_weight = None, labels = None, y_true, y_pred, normalize = None):
		self.y_pred = y_pred
		self.normalize = normalize
		Metrics.__init__(self, sample_weight, labels, y_true)
		self.value = CM(sample_weight = self.sample_weight,
			normalize = self.normalize,
			y_pred = self.y_pred,
			labels = self.labels,
			y_true = self.y_true)

