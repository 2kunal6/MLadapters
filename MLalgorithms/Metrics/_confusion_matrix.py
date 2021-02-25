
from sklearn.metrics import confusion_matrix as CM
from MLalgorithms._Metrics import Metrics


class confusion_matrix(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, sample_weight=None, normalize=None):
		self.y_pred = y_pred
<<<<<<< HEAD
		Metrics.__init__(self, labels=labels, y_true=y_true, sample_weight=sample_weight)
		self.value = CM(sample_weight = self.sample_weight,
			labels = self.labels,
			normalize = self.normalize,
			y_pred = self.y_pred,
			y_true = self.y_true)
=======
		self.normalize = normalize
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = CM(normalize = self.normalize,
			y_true = self.y_true,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			labels = self.labels)
>>>>>>> 0430e16ab26f6fca7bf3d07a8f46b7265698b0cb

