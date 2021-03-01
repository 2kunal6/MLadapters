
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.average = average
		self.y_pred = y_pred
		self.pos_label = pos_label
		self.zero_division = zero_division
		Metrics.__init__(self, y_true=y_true, sample_weight=sample_weight, labels=labels)
		self.value = JS(y_pred = self.y_pred,
			average = self.average,
			zero_division = self.zero_division,
			labels = self.labels,
			sample_weight = self.sample_weight,
			pos_label = self.pos_label,
			y_true = self.y_true)

