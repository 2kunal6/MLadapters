
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.y_pred = y_pred
		self.average = average
		self.pos_label = pos_label
		self.zero_division = zero_division
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = JS(y_true = self.y_true,
			sample_weight = self.sample_weight,
			pos_label = self.pos_label,
			y_pred = self.y_pred,
			labels = self.labels,
			average = self.average,
			zero_division = self.zero_division)

