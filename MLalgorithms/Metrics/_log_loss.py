
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, y_pred, eps=1e-15, labels=None, normalize=True, sample_weight=None):
		self.y_pred = y_pred
		self.normalize = normalize
		self.eps = eps
<<<<<<< HEAD
		Metrics.__init__(self, labels=labels, y_true=y_true, sample_weight=sample_weight)
		self.value = LL(sample_weight = self.sample_weight,
			labels = self.labels,
			normalize = self.normalize,
			eps = self.eps,
			y_pred = self.y_pred,
			y_true = self.y_true)
=======
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = LL(normalize = self.normalize,
			y_true = self.y_true,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			labels = self.labels,
			eps = self.eps)
>>>>>>> 0430e16ab26f6fca7bf3d07a8f46b7265698b0cb

