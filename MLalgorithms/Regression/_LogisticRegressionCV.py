
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
		self.refit = refit
		self.tol = tol
		self.verbose = verbose
		self.class_weight = class_weight
		self.multi_class = multi_class
		self.max_iter = max_iter
		self.dual = dual
		self.Cs = Cs
		self.solver = solver
		self.n_jobs = n_jobs
		self.scoring = scoring
		self.intercept_scaling = intercept_scaling
		self.cv = cv
		self.penalty = penalty
		self.random_state = random_state
		self.l1_ratios = l1_ratios
		self.fit_intercept = fit_intercept
		self.model = LRCV(dual = self.dual,
			multi_class = self.multi_class,
			max_iter = self.max_iter,
			scoring = self.scoring,
			verbose = self.verbose,
			class_weight = self.class_weight,
			intercept_scaling = self.intercept_scaling,
			cv = self.cv,
			random_state = self.random_state,
			refit = self.refit,
			tol = self.tol,
			l1_ratios = self.l1_ratios,
			solver = self.solver,
			n_jobs = self.n_jobs,
			Cs = self.Cs,
			fit_intercept = self.fit_intercept,
			penalty = self.penalty)

