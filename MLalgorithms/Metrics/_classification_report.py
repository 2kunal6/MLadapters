
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.digits = digits
		self.y_pred = y_pred
		self.target_names = target_names
		self.zero_division = zero_division
		self.output_dict = output_dict
		Metrics.__init__(self, sample_weight=sample_weight, y_true=y_true, labels=labels)
		self.value = CR(target_names = self.target_names,
			y_true = self.y_true,
			digits = self.digits,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			output_dict = self.output_dict,
			labels = self.labels,
			zero_division = self.zero_division)

