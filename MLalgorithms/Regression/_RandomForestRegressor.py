
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor as RFR
from MLalgorithms._Regression import Regression


class RandomForestRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, fit_intercept=True, normalize=False, copy_X=True):
		self.bootstrap = bootstrap
		self.min_samples_split = min_samples_split
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.oob_score = oob_score
		self.max_leaf_nodes = max_leaf_nodes
		self.warm_start = warm_start
		self.verbose = verbose
		self.ccp_alpha = ccp_alpha
		self.min_impurity_decrease = min_impurity_decrease
		self.random_state = random_state
		self.n_estimators = n_estimators
		self.min_samples_leaf = min_samples_leaf
		self.max_samples = max_samples
		self.n_jobs = n_jobs
		self.max_features = max_features
		self.min_impurity_split = min_impurity_split
		self.max_depth = max_depth
		self.criterion = criterion
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = RFR(fit_intercept = self.fit_intercept,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			min_impurity_decrease = self.min_impurity_decrease,
			max_depth = self.max_depth,
			bootstrap = self.bootstrap,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			ccp_alpha = self.ccp_alpha,
			min_samples_split = self.min_samples_split,
			min_samples_leaf = self.min_samples_leaf,
			oob_score = self.oob_score,
			max_leaf_nodes = self.max_leaf_nodes,
			n_estimators = self.n_estimators,
			random_state = self.random_state,
			n_jobs = self.n_jobs,
			max_samples = self.max_samples,
			max_features = self.max_features,
			normalize = self.normalize,
			min_impurity_split = self.min_impurity_split,
			verbose = self.verbose,
			criterion = self.criterion)

	def predict(self, X):
		return self.model.predict(X=X)

