
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, y_pred, eps=1e-15, labels=None, normalize=True, sample_weight=None):
		self.normalize = normalize
		self.eps = eps
		self.y_pred = y_pred
		Metrics.__init__(self, y_true=y_true, sample_weight=sample_weight, labels=labels)
		self.value = LL(y_pred = self.y_pred,
			eps = self.eps,
			normalize = self.normalize,
			labels = self.labels,
			sample_weight = self.sample_weight,
			y_true = self.y_true)

