
from sklearn.metrics import log_loss as LL
from MLalgorithms.Metrics import Metrics


class log_loss(Metrics):
	
	def __init__(self, y_true, labels, sample_weight, eps = 1e-15, normalize = True, y_pred):
		self.eps = eps
		self.normalize = normalize
		self.y_pred = y_pred
		Metrics.__init__(self, y_true, labels, sample_weight)
		self.value = LL(sample_weight = self.sample_weight,
			y_true = self.y_true,
			y_pred = self.y_pred,
			labels = self.labels,
			normalize = self.normalize,
			eps = self.eps)

