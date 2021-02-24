
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.zero_division = zero_division
		self.y_pred = y_pred
		self.target_names = target_names
		self.digits = digits
		self.output_dict = output_dict
		Metrics.__init__(self, labels=labels, y_true=y_true, sample_weight=sample_weight)
		self.value = CR(y_pred = self.y_pred,
			y_true = self.y_true,
			target_names = self.target_names,
			sample_weight = self.sample_weight,
			zero_division = self.zero_division,
			labels = self.labels,
			digits = self.digits,
			output_dict = self.output_dict)

