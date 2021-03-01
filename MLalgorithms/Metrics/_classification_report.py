
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.zero_division = zero_division
		self.output_dict = output_dict
		self.digits = digits
		self.y_pred = y_pred
		self.target_names = target_names
		Metrics.__init__(self, y_true=y_true, labels=labels, sample_weight=sample_weight)
		self.value = CR(y_true = self.y_true,
			zero_division = self.zero_division,
			labels = self.labels,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			target_names = self.target_names,
			output_dict = self.output_dict,
			digits = self.digits)

