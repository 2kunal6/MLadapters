
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.pos_label = pos_label
		self.y_pred = y_pred
		self.zero_division = zero_division
		self.average = average
		Metrics.__init__(self, labels=labels, y_true=y_true, sample_weight=sample_weight)
		self.value = JS(y_pred = self.y_pred,
			y_true = self.y_true,
			pos_label = self.pos_label,
			sample_weight = self.sample_weight,
			zero_division = self.zero_division,
			labels = self.labels,
			average = self.average)

