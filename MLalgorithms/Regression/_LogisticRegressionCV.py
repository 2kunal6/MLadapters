
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.class_weight = class_weight
		self.verbose = verbose
		self.solver = solver
		self.tol = tol
		self.cv = cv
		self.intercept_scaling = intercept_scaling
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.l1_ratios = l1_ratios
		self.fit_intercept = fit_intercept
		self.penalty = penalty
		self.refit = refit
		self.multi_class = multi_class
		self.scoring = scoring
		self.Cs = Cs
		self.random_state = random_state
		self.dual = dual
		self.model = LRCV(scoring = self.scoring,
			Cs = self.Cs,
			verbose = self.verbose,
			penalty = self.penalty,
			dual = self.dual,
			cv = self.cv,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			refit = self.refit,
			intercept_scaling = self.intercept_scaling,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			multi_class = self.multi_class,
			random_state = self.random_state,
			solver = self.solver,
			l1_ratios = self.l1_ratios,
			n_jobs = self.n_jobs)

