
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.tol = tol
		self.multi_class = multi_class
		self.intercept_scaling = intercept_scaling
		self.Cs = Cs
		self.n_jobs = n_jobs
		self.scoring = scoring
		self.verbose = verbose
		self.class_weight = class_weight
		self.refit = refit
		self.l1_ratios = l1_ratios
		self.dual = dual
		self.random_state = random_state
		self.cv = cv
		self.solver = solver
		self.fit_intercept = fit_intercept
		self.penalty = penalty
		self.max_iter = max_iter
		self.model = LRCV(dual = self.dual,
			fit_intercept = self.fit_intercept,
			refit = self.refit,
			intercept_scaling = self.intercept_scaling,
			random_state = self.random_state,
			max_iter = self.max_iter,
			l1_ratios = self.l1_ratios,
			n_jobs = self.n_jobs,
			multi_class = self.multi_class,
			solver = self.solver,
			penalty = self.penalty,
			tol = self.tol,
			class_weight = self.class_weight,
			scoring = self.scoring,
			verbose = self.verbose,
			cv = self.cv,
			Cs = self.Cs)

