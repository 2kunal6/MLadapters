
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, y_pred, eps=1e-15, labels=None, normalize=True, sample_weight=None):
		self.normalize = normalize
		self.y_pred = y_pred
		self.eps = eps
		Metrics.__init__(self, labels=labels, sample_weight=sample_weight, y_true=y_true)
		self.value = LL(y_true = self.y_true,
			y_pred = self.y_pred,
			labels = self.labels,
			eps = self.eps,
			sample_weight = self.sample_weight,
			normalize = self.normalize)

