
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.verbose = verbose
		self.multi_class = multi_class
		self.Cs = Cs
		self.intercept_scaling = intercept_scaling
		self.scoring = scoring
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.refit = refit
		self.tol = tol
		self.solver = solver
		self.l1_ratios = l1_ratios
		self.random_state = random_state
		self.cv = cv
		self.penalty = penalty
		self.dual = dual
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.model = LRCV(refit = self.refit,
			random_state = self.random_state,
			tol = self.tol,
			solver = self.solver,
			class_weight = self.class_weight,
			n_jobs = self.n_jobs,
			dual = self.dual,
			penalty = self.penalty,
			scoring = self.scoring,
			Cs = self.Cs,
			intercept_scaling = self.intercept_scaling,
			verbose = self.verbose,
			fit_intercept = self.fit_intercept,
			l1_ratios = self.l1_ratios,
			cv = self.cv,
			max_iter = self.max_iter,
			multi_class = self.multi_class)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

