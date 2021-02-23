
from sklearn.metrics import hinge_loss as HL
from MLalgorithms._Metrics import Metrics


class hinge_loss(Metrics):
	
	def __init__(self, sample_weight = None, labels = None, y_true, pred_decision):
		self.pred_decision = pred_decision
		Metrics.__init__(self, sample_weight, labels, y_true)
		self.value = HL(y_true = self.y_true,
			labels = self.labels,
			sample_weight = self.sample_weight,
			pred_decision = self.pred_decision)

