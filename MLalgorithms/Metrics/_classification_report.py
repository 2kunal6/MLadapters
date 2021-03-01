
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.zero_division = zero_division
		self.digits = digits
		self.output_dict = output_dict
		self.y_pred = y_pred
		self.target_names = target_names
		Metrics.__init__(self, labels=labels, sample_weight=sample_weight, y_true=y_true)
		self.value = CR(y_pred = self.y_pred,
			output_dict = self.output_dict,
			digits = self.digits,
			target_names = self.target_names,
			zero_division = self.zero_division,
			y_true = self.y_true,
			sample_weight = self.sample_weight,
			labels = self.labels)

