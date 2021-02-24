
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.pos_label = pos_label
		self.y_pred = y_pred
		self.average = average
		self.zero_division = zero_division
		Metrics.__init__(self, y_true=y_true, labels=labels, sample_weight=sample_weight)
		self.value = JS(zero_division = self.zero_division,
			pos_label = self.pos_label,
			labels = self.labels,
			y_pred = self.y_pred,
			y_true = self.y_true,
			sample_weight = self.sample_weight,
			average = self.average)

