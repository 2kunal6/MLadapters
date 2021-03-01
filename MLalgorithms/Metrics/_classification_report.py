
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'):
		self.output_dict = output_dict
		self.digits = digits
		self.y_pred = y_pred
		self.zero_division = zero_division
		self.target_names = target_names
		Metrics.__init__(self, labels=labels, y_true=y_true, sample_weight=sample_weight)
		self.value = CR(zero_division = self.zero_division,
			y_pred = self.y_pred,
			output_dict = self.output_dict,
			sample_weight = self.sample_weight,
			target_names = self.target_names,
			labels = self.labels,
			y_true = self.y_true,
			digits = self.digits)

