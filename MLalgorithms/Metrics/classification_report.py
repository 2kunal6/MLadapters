
from sklearn.metrics import classification_report as CR
from MLalgorithms.Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, y_true, labels, sample_weight, zero_division = 'warn', y_pred, output_dict = False, digits = 2, target_names = None):
		self.zero_division = zero_division
		self.y_pred = y_pred
		self.output_dict = output_dict
		self.digits = digits
		self.target_names = target_names
		Metrics.__init__(self, y_true, labels, sample_weight)
		self.value = CR(sample_weight = self.sample_weight,
			y_true = self.y_true,
			zero_division = self.zero_division,
			y_pred = self.y_pred,
			labels = self.labels,
			digits = self.digits,
			target_names = self.target_names,
			output_dict = self.output_dict)

