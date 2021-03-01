
from sklearn.ensemble import GradientBoostingRegressor as GBR
from MLalgorithms._Regression import Regression


class GradientBoostingRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, monitor=None):
		return self.model.fit(y=y,
			monitor=monitor,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
		self.min_samples_leaf = min_samples_leaf
		self.min_impurity_split = min_impurity_split
		self.validation_fraction = validation_fraction
		self.learning_rate = learning_rate
		self.tol = tol
		self.warm_start = warm_start
		self.subsample = subsample
		self.n_iter_no_change = n_iter_no_change
		self.criterion = criterion
		self.verbose = verbose
		self.max_leaf_nodes = max_leaf_nodes
		self.random_state = random_state
		self.n_estimators = n_estimators
		self.min_samples_split = min_samples_split
		self.alpha = alpha
		self.min_impurity_decrease = min_impurity_decrease
		self.max_features = max_features
		self.ccp_alpha = ccp_alpha
		self.loss = loss
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.max_depth = max_depth
		self.model = GBR(n_iter_no_change = self.n_iter_no_change,
			warm_start = self.warm_start,
			min_impurity_split = self.min_impurity_split,
			alpha = self.alpha,
			ccp_alpha = self.ccp_alpha,
			verbose = self.verbose,
			random_state = self.random_state,
			loss = self.loss,
			max_depth = self.max_depth,
			learning_rate = self.learning_rate,
			tol = self.tol,
			min_samples_leaf = self.min_samples_leaf,
			max_leaf_nodes = self.max_leaf_nodes,
			min_samples_split = self.min_samples_split,
			max_features = self.max_features,
			min_impurity_decrease = self.min_impurity_decrease,
			subsample = self.subsample,
			validation_fraction = self.validation_fraction,
			criterion = self.criterion,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			n_estimators = self.n_estimators)

