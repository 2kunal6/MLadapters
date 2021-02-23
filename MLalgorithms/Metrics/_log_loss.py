
from sklearn.metrics import log_loss as LL
from MLalgorithms._Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, sample_weight = None, labels = None, y_true, normalize = True, y_pred, eps = 1e-15):
		self.normalize = normalize
		self.y_pred = y_pred
		self.eps = eps
		Metrics.__init__(self, sample_weight, labels, y_true)
		self.value = LL(sample_weight = self.sample_weight,
			normalize = self.normalize,
			y_pred = self.y_pred,
			eps = self.eps,
			labels = self.labels,
			y_true = self.y_true)

