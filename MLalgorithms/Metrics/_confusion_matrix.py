
from sklearn.metrics import confusion_matrix as CM
from MLalgorithms._Metrics import Metrics


class confusion_matrix(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, sample_weight=None, normalize=None):
		self.y_pred = y_pred
		self.normalize = normalize
		Metrics.__init__(self, sample_weight=sample_weight, labels=labels, y_true=y_true)
		self.value = CM(y_true = self.y_true,
			normalize = self.normalize,
			y_pred = self.y_pred,
			labels = self.labels,
			sample_weight = self.sample_weight)

