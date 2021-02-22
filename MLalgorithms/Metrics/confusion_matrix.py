
from sklearn.metrics import confusion_matrix as CM
from MLalgorithms.Metrics import Metrics


class confusion_matrix(Metrics):
	
	def __init__(self, y_true, labels, sample_weight, y_pred, normalize = None):
		self.y_pred = y_pred
		self.normalize = normalize
		Metrics.__init__(self, y_true, labels, sample_weight)
		self.value = CM(sample_weight = self.sample_weight,
			y_true = self.y_true,
			y_pred = self.y_pred,
			labels = self.labels,
			normalize = self.normalize)

