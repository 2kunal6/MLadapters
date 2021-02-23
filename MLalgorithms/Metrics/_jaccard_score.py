
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, sample_weight = None, labels = None, y_true, y_pred, pos_label = 1, zero_division = 'warn', average = 'binary'):
		self.y_pred = y_pred
		self.pos_label = pos_label
		self.zero_division = zero_division
		self.average = average
		Metrics.__init__(self, sample_weight, labels, y_true)
		self.value = JS(zero_division = self.zero_division,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			labels = self.labels,
			average = self.average,
			y_true = self.y_true,
			pos_label = self.pos_label)

