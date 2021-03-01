
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor as RFR
from MLalgorithms._Regression import Regression


class RandomForestRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
		self.verbose = verbose
		self.n_estimators = n_estimators
		self.ccp_alpha = ccp_alpha
		self.max_depth = max_depth
		self.warm_start = warm_start
		self.min_impurity_split = min_impurity_split
		self.min_samples_leaf = min_samples_leaf
		self.max_features = max_features
		self.min_samples_split = min_samples_split
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.n_jobs = n_jobs
		self.max_leaf_nodes = max_leaf_nodes
		self.random_state = random_state
		self.max_samples = max_samples
		self.min_impurity_decrease = min_impurity_decrease
		self.oob_score = oob_score
		self.criterion = criterion
		self.bootstrap = bootstrap
		self.model = RFR(criterion = self.criterion,
			max_leaf_nodes = self.max_leaf_nodes,
			min_samples_split = self.min_samples_split,
			min_samples_leaf = self.min_samples_leaf,
			random_state = self.random_state,
			min_impurity_split = self.min_impurity_split,
			warm_start = self.warm_start,
			min_impurity_decrease = self.min_impurity_decrease,
			max_samples = self.max_samples,
			ccp_alpha = self.ccp_alpha,
			max_features = self.max_features,
			n_jobs = self.n_jobs,
			max_depth = self.max_depth,
			verbose = self.verbose,
			n_estimators = self.n_estimators,
			bootstrap = self.bootstrap,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			oob_score = self.oob_score)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

