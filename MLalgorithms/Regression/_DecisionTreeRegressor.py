
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor as DTR
from MLalgorithms._Regression import Regression


class DecisionTreeRegressor(Regression):
	
	def __init__(self, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0):
		self.max_leaf_nodes = max_leaf_nodes
		self.min_impurity_split = min_impurity_split
		self.random_state = random_state
		self.ccp_alpha = ccp_alpha
		self.max_features = max_features
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.splitter = splitter
		self.min_samples_leaf = min_samples_leaf
		self.min_impurity_decrease = min_impurity_decrease
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.criterion = criterion
		self.model = DTR(min_samples_leaf = self.min_samples_leaf,
			max_leaf_nodes = self.max_leaf_nodes,
			random_state = self.random_state,
			criterion = self.criterion,
			ccp_alpha = self.ccp_alpha,
			min_samples_split = self.min_samples_split,
			max_features = self.max_features,
			splitter = self.splitter,
			min_impurity_decrease = self.min_impurity_decrease,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			max_depth = self.max_depth,
			min_impurity_split = self.min_impurity_split)

	def predict(self, X, check_input=True):
		return self.model.predict(check_input=check_input,
			X=X)

	def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'):
		return self.model.fit(X=X,
			X_idx_sorted=X_idx_sorted,
			y=y,
			check_input=check_input,
			sample_weight=sample_weight)

