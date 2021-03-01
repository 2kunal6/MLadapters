
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor as GBR
from MLalgorithms._Regression import Regression


class GradientBoostingRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None, monitor=None):
		return self.model.fit(y=y,
			monitor=monitor,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0, fit_intercept=True, normalize=False, copy_X=True):
		self.min_samples_split = min_samples_split
		self.n_iter_no_change = n_iter_no_change
		self.verbose = verbose
		self.criterion = criterion
		self.ccp_alpha = ccp_alpha
		self.n_estimators = n_estimators
		self.min_samples_leaf = min_samples_leaf
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.validation_fraction = validation_fraction
		self.random_state = random_state
		self.loss = loss
		self.subsample = subsample
		self.max_features = max_features
		self.warm_start = warm_start
		self.max_leaf_nodes = max_leaf_nodes
		self.tol = tol
		self.max_depth = max_depth
		self.min_impurity_split = min_impurity_split
		self.learning_rate = learning_rate
		self.alpha = alpha
		self.min_impurity_decrease = min_impurity_decrease
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = GBR(validation_fraction = self.validation_fraction,
			fit_intercept = self.fit_intercept,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			min_impurity_decrease = self.min_impurity_decrease,
			max_depth = self.max_depth,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			ccp_alpha = self.ccp_alpha,
			min_samples_split = self.min_samples_split,
			min_samples_leaf = self.min_samples_leaf,
			loss = self.loss,
			max_leaf_nodes = self.max_leaf_nodes,
			n_estimators = self.n_estimators,
			n_iter_no_change = self.n_iter_no_change,
			alpha = self.alpha,
			random_state = self.random_state,
			learning_rate = self.learning_rate,
			subsample = self.subsample,
			max_features = self.max_features,
			normalize = self.normalize,
			min_impurity_split = self.min_impurity_split,
			verbose = self.verbose,
			tol = self.tol,
			criterion = self.criterion)

	def predict(self, X):
		return self.model.predict(X=X)

