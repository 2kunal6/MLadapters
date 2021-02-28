
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, y_pred, eps=1e-15, labels=None, normalize=True, sample_weight=None):
		self.eps = eps
		self.y_pred = y_pred
		self.normalize = normalize
		Metrics.__init__(self, sample_weight=sample_weight, labels=labels, y_true=y_true)
		self.value = LL(labels = self.labels,
			y_true = self.y_true,
			y_pred = self.y_pred,
			eps = self.eps,
			normalize = self.normalize,
			sample_weight = self.sample_weight)

