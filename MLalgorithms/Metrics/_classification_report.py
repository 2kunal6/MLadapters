
from sklearn.metrics import classification_report as CR
from MLalgorithms._Metrics import Metrics


class classification_report(Metrics):
	
	def __init__(self, sample_weight = None, labels = None, y_true, output_dict = False, target_names = None, y_pred, zero_division = 'warn', digits = 2):
		self.output_dict = output_dict
		self.target_names = target_names
		self.y_pred = y_pred
		self.zero_division = zero_division
		self.digits = digits
		Metrics.__init__(self, sample_weight, labels, y_true)
		self.value = CR(zero_division = self.zero_division,
			output_dict = self.output_dict,
			sample_weight = self.sample_weight,
			y_pred = self.y_pred,
			labels = self.labels,
			target_names = self.target_names,
			y_true = self.y_true,
			digits = self.digits)

