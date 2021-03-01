
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.solver = solver
		self.scoring = scoring
		self.dual = dual
		self.multi_class = multi_class
		self.Cs = Cs
		self.random_state = random_state
		self.verbose = verbose
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.refit = refit
		self.tol = tol
		self.l1_ratios = l1_ratios
		self.cv = cv
		self.penalty = penalty
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.intercept_scaling = intercept_scaling
		self.model = LRCV(random_state = self.random_state,
			multi_class = self.multi_class,
			Cs = self.Cs,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			class_weight = self.class_weight,
			penalty = self.penalty,
			max_iter = self.max_iter,
			scoring = self.scoring,
			dual = self.dual,
			intercept_scaling = self.intercept_scaling,
			verbose = self.verbose,
			solver = self.solver,
			cv = self.cv,
			refit = self.refit,
			l1_ratios = self.l1_ratios,
			n_jobs = self.n_jobs)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

