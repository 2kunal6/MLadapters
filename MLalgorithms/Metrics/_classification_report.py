
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.y_pred = y_pred
		self.target_names = target_names
		self.digits = digits
		self.output_dict = output_dict
		self.zero_division = zero_division
		Metrics.__init__(self, sample_weight=sample_weight, labels=labels, y_true=y_true)
		self.value = CR(target_names = self.target_names,
			labels = self.labels,
			zero_division = self.zero_division,
			output_dict = self.output_dict,
			y_true = self.y_true,
			y_pred = self.y_pred,
			digits = self.digits,
			sample_weight = self.sample_weight)

