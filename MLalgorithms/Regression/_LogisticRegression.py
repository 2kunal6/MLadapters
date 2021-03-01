
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as LR
from MLalgorithms._Regression import Regression


class LogisticRegression(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, normalize=False, copy_X=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
		self.C = C
		self.penalty = penalty
		self.warm_start = warm_start
		self.verbose = verbose
		self.class_weight = class_weight
		self.max_iter = max_iter
		self.l1_ratio = l1_ratio
		self.solver = solver
		self.tol = tol
		self.random_state = random_state
		self.multi_class = multi_class
		self.n_jobs = n_jobs
		self.intercept_scaling = intercept_scaling
		self.dual = dual
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = LR(class_weight = self.class_weight,
			penalty = self.penalty,
			multi_class = self.multi_class,
			l1_ratio = self.l1_ratio,
			warm_start = self.warm_start,
			C = self.C,
			intercept_scaling = self.intercept_scaling,
			dual = self.dual,
			copy_X = self.copy_X,
			normalize = self.normalize,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			n_jobs = self.n_jobs,
			verbose = self.verbose,
			max_iter = self.max_iter,
			tol = self.tol,
			solver = self.solver)

	def predict(self, X):
		return self.model.predict(X=X)

