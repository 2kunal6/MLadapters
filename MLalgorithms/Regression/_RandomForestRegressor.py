
from sklearn.ensemble import RandomForestRegressor as RFR
from MLalgorithms._Regression import Regression


class RandomForestRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
		self.oob_score = oob_score
		self.min_samples_leaf = min_samples_leaf
		self.warm_start = warm_start
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.n_jobs = n_jobs
		self.min_impurity_decrease = min_impurity_decrease
		self.max_samples = max_samples
		self.min_impurity_split = min_impurity_split
		self.verbose = verbose
		self.bootstrap = bootstrap
		self.max_leaf_nodes = max_leaf_nodes
		self.criterion = criterion
		self.max_depth = max_depth
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.random_state = random_state
		self.ccp_alpha = ccp_alpha
		self.min_samples_split = min_samples_split
		self.model = RFR(min_samples_leaf = self.min_samples_leaf,
			random_state = self.random_state,
			warm_start = self.warm_start,
			min_impurity_split = self.min_impurity_split,
			max_features = self.max_features,
			bootstrap = self.bootstrap,
			ccp_alpha = self.ccp_alpha,
			max_depth = self.max_depth,
			n_jobs = self.n_jobs,
			max_leaf_nodes = self.max_leaf_nodes,
			min_weight_fraction_leaf = self.min_weight_fraction_leaf,
			oob_score = self.oob_score,
			criterion = self.criterion,
			min_samples_split = self.min_samples_split,
			verbose = self.verbose,
			n_estimators = self.n_estimators,
			max_samples = self.max_samples,
			min_impurity_decrease = self.min_impurity_decrease)

