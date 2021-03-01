
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.digits = digits
		self.y_pred = y_pred
		self.zero_division = zero_division
		self.target_names = target_names
		self.output_dict = output_dict
		Metrics.__init__(self, y_true=y_true, sample_weight=sample_weight, labels=labels)
		self.value = CR(y_pred = self.y_pred,
			zero_division = self.zero_division,
			target_names = self.target_names,
			y_true = self.y_true,
			labels = self.labels,
			sample_weight = self.sample_weight,
			output_dict = self.output_dict,
			digits = self.digits)

