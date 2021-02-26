
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.average = average
		self.zero_division = zero_division
		self.y_pred = y_pred
		self.pos_label = pos_label
		Metrics.__init__(self, y_true=y_true, sample_weight=sample_weight, labels=labels)
		self.value = JS(y_pred = self.y_pred,
			pos_label = self.pos_label,
			y_true = self.y_true,
			average = self.average,
			labels = self.labels,
			zero_division = self.zero_division,
			sample_weight = self.sample_weight)

