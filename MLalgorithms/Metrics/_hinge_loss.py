
from sklearn.metrics import hinge_loss as HL
from MLalgorithms._Metrics import Metrics


class hinge_loss(Metrics):
	
	def __init__(self, y_true, pred_decision, labels=None, sample_weight=None):
		self.pred_decision = pred_decision
<<<<<<< HEAD
		Metrics.__init__(self, labels=labels, y_true=y_true, sample_weight=sample_weight)
		self.value = HL(labels = self.labels,
			pred_decision = self.pred_decision,
			sample_weight = self.sample_weight,
			y_true = self.y_true)
=======
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = HL(sample_weight = self.sample_weight,
			labels = self.labels,
			y_true = self.y_true,
			pred_decision = self.pred_decision)
>>>>>>> 0430e16ab26f6fca7bf3d07a8f46b7265698b0cb

