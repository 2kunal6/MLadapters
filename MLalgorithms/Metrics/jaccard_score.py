
from sklearn.metrics import jaccard_score as JS
from MLalgorithms.Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, labels, sample_weight, y_pred, pos_label = 1, average = 'binary', zero_division = 'warn'):
		self.y_pred = y_pred
		self.pos_label = pos_label
		self.average = average
		self.zero_division = zero_division
		Metrics.__init__(self, y_true, labels, sample_weight)
		self.value = JS(sample_weight = self.sample_weight,
			average = self.average,
			y_pred = self.y_pred,
			y_true = self.y_true,
			zero_division = self.zero_division,
			labels = self.labels,
			pos_label = self.pos_label)

