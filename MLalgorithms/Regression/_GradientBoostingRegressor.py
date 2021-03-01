
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor as GBR
from MLalgorithms._Regression import Regression


class GradientBoostingRegressor(Regression):
	
	def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
		self.min_impurity_split = min_impurity_split
		self.learning_rate = learning_rate
		self.min_samples_split = min_samples_split
		self.n_estimators = n_estimators
		self.min_samples_leaf = min_samples_leaf
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.criterion = criterion
		self.random_state = random_state
		self.verbose = verbose
		self.alpha = alpha
		self.validation_fraction = validation_fraction
		self.max_leaf_nodes = max_leaf_nodes
		self.max_depth = max_depth
		self.max_features = max_features
		self.tol = tol
		self.min_impurity_decrease = min_impurity_decrease
		self.ccp_alpha = ccp_alpha
		self.loss = loss
		self.subsample = subsample
		self.n_iter_no_change = n_iter_no_change
		self.warm_start = warm_start
		self.model = GBR(random_state = self.random_state,
			min_samples_split = self.min_samples_split,
			learning_rate = self.learning_rate,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			verbose = self.verbose,
			validation_fraction = self.validation_fraction,
			loss = self.loss,
			min_samples_leaf = self.min_samples_leaf,
			n_iter_no_change = self.n_iter_no_change,
			criterion = self.criterion,
			ccp_alpha = self.ccp_alpha,
			alpha = self.alpha,
			min_impurity_split = self.min_impurity_split,
			n_estimators = self.n_estimators,
			max_features = self.max_features,
			tol = self.tol,
			subsample = self.subsample,
			max_leaf_nodes = self.max_leaf_nodes,
			min_impurity_decrease = self.min_impurity_decrease,
			warm_start = self.warm_start,
			max_depth = self.max_depth)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, monitor=None):
		return self.model.fit(y=y,
			monitor=monitor,
			sample_weight=sample_weight,
			X=X)

