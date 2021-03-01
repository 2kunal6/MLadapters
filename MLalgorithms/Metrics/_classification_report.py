
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.y_pred = y_pred
		self.digits = digits
		self.zero_division = zero_division
		self.target_names = target_names
		self.output_dict = output_dict
		Metrics.__init__(self, labels=labels, sample_weight=sample_weight, y_true=y_true)
		self.value = CR(y_pred = self.y_pred,
			digits = self.digits,
			output_dict = self.output_dict,
			zero_division = self.zero_division,
			y_true = self.y_true,
			labels = self.labels,
			sample_weight = self.sample_weight,
			target_names = self.target_names)

