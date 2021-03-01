
from sklearn.tree import DecisionTreeClassifier as DTC
from MLalgorithms._Classification import Classification


class DecisionTreeClassifier(Classification):
	
	def predict(self, X, check_input=True):
		return self.model.predict(check_input=check_input,
			X=X)

	def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'):
		return self.model.fit(check_input=check_input,
			X=X,
			y=y,
			X_idx_sorted=X_idx_sorted,
			sample_weight=sample_weight)

	def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0):
		self.min_samples_leaf = min_samples_leaf
		self.min_impurity_decrease = min_impurity_decrease
		self.splitter = splitter
		self.min_impurity_split = min_impurity_split
		self.class_weight = class_weight
		self.max_features = max_features
		self.max_leaf_nodes = max_leaf_nodes
		self.max_depth = max_depth
		self.criterion = criterion
		self.random_state = random_state
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.ccp_alpha = ccp_alpha
		self.min_samples_split = min_samples_split
		self.model = DTC(min_samples_leaf = self.min_samples_leaf,
			min_impurity_split = self.min_impurity_split,
			min_impurity_decrease = self.min_impurity_decrease,
			ccp_alpha = self.ccp_alpha,
			max_depth = self.max_depth,
			max_leaf_nodes = self.max_leaf_nodes,
			criterion = self.criterion,
			class_weight = self.class_weight,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			min_samples_split = self.min_samples_split,
			random_state = self.random_state,
			max_features = self.max_features,
			splitter = self.splitter)

