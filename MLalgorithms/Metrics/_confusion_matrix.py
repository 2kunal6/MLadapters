
from sklearn.metrics import confusion_matrix as CM
from MLalgorithms._Metrics import Metrics


class confusion_matrix(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, sample_weight=None, normalize=None):
		self.normalize = normalize
		self.y_pred = y_pred
		Metrics.__init__(self, y_true=y_true, labels=labels, sample_weight=sample_weight)
		self.value = CM(normalize = self.normalize,
			y_pred = self.y_pred,
			y_true = self.y_true,
			sample_weight = self.sample_weight,
			labels = self.labels)

