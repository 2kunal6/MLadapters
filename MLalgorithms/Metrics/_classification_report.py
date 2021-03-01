
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.target_names = target_names
		self.digits = digits
		self.y_pred = y_pred
		self.zero_division = zero_division
		self.output_dict = output_dict
		Metrics.__init__(self, labels=labels, sample_weight=sample_weight, y_true=y_true)
		self.value = CR(output_dict = self.output_dict,
			zero_division = self.zero_division,
			y_true = self.y_true,
			digits = self.digits,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			labels = self.labels,
			target_names = self.target_names)

