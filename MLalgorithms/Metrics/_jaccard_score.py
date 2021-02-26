
from sklearn.metrics import jaccard_score as JS
from MLalgorithms._Metrics import Metrics


class jaccard_score(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
		self.zero_division = zero_division
		self.pos_label = pos_label
		self.y_pred = y_pred
		self.average = average
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = JS(sample_weight = self.sample_weight,
			pos_label = self.pos_label,
			labels = self.labels,
			y_pred = self.y_pred,
			average = self.average,
			y_true = self.y_true,
			zero_division = self.zero_division)

