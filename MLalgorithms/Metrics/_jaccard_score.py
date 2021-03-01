
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.y_pred = y_pred
		self.average = average
		self.zero_division = zero_division
		self.pos_label = pos_label
		Metrics.__init__(self, y_true=y_true, sample_weight=sample_weight, labels=labels)
		self.value = JS(average = self.average,
			pos_label = self.pos_label,
			y_true = self.y_true,
			labels = self.labels,
			y_pred = self.y_pred,
			sample_weight = self.sample_weight,
			zero_division = self.zero_division)

