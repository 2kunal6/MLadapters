
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, y_pred, eps=1e-15, labels=None, normalize=True, sample_weight=None):
		self.normalize = normalize
		self.y_pred = y_pred
		self.eps = eps
		Metrics.__init__(self, y_true=y_true, labels=labels, sample_weight=sample_weight)
		self.value = LL(eps = self.eps,
			labels = self.labels,
			normalize = self.normalize,
			y_pred = self.y_pred,
			sample_weight = self.sample_weight,
			y_true = self.y_true)

