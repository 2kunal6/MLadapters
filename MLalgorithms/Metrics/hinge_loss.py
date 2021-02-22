
from sklearn.metrics import hinge_loss as HL
from MLalgorithms.Metrics import Metrics


class hinge_loss(Metrics):
	
	def __init__(self, y_true, labels, sample_weight, pred_decision):
		self.pred_decision = pred_decision
		Metrics.__init__(self, y_true, labels, sample_weight)
		self.value = HL(pred_decision = self.pred_decision,
			y_true = self.y_true,
			sample_weight = self.sample_weight,
			labels = self.labels)

