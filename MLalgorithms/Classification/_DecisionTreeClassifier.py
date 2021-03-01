
from sklearn.tree import DecisionTreeClassifier as DTC
from MLalgorithms._Classification import Classification


class DecisionTreeClassifier(Classification):
	
	def predict(self, X, check_input=True):
		return self.model.predict(check_input=check_input,
			X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			check_input=check_input,
			X=X)

	def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0):
		self.class_weight = class_weight
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.min_samples_leaf = min_samples_leaf
		self.max_depth = max_depth
		self.criterion = criterion
		self.min_impurity_split = min_impurity_split
		self.splitter = splitter
		self.min_impurity_decrease = min_impurity_decrease
		self.random_state = random_state
		self.max_leaf_nodes = max_leaf_nodes
		self.min_samples_split = min_samples_split
		self.max_features = max_features
		self.ccp_alpha = ccp_alpha
		self.model = DTC(ccp_alpha = self.ccp_alpha,
			max_features = self.max_features,
			max_depth = self.max_depth,
			criterion = self.criterion,
			class_weight = self.class_weight,
			splitter = self.splitter,
			random_state = self.random_state,
			min_samples_leaf = self.min_samples_leaf,
			min_impurity_decrease = self.min_impurity_decrease,
			max_leaf_nodes = self.max_leaf_nodes,
			min_samples_split = self.min_samples_split,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			min_impurity_split = self.min_impurity_split)

