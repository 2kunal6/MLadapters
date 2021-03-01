
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.multi_class = multi_class
		self.random_state = random_state
		self.scoring = scoring
		self.class_weight = class_weight
		self.solver = solver
		self.n_jobs = n_jobs
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.verbose = verbose
		self.l1_ratios = l1_ratios
		self.tol = tol
		self.refit = refit
		self.cv = cv
		self.intercept_scaling = intercept_scaling
		self.penalty = penalty
		self.Cs = Cs
		self.dual = dual
		self.model = LRCV(scoring = self.scoring,
			tol = self.tol,
			Cs = self.Cs,
			penalty = self.penalty,
			dual = self.dual,
			n_jobs = self.n_jobs,
			random_state = self.random_state,
			l1_ratios = self.l1_ratios,
			verbose = self.verbose,
			cv = self.cv,
			multi_class = self.multi_class,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			solver = self.solver,
			fit_intercept = self.fit_intercept,
			refit = self.refit,
			intercept_scaling = self.intercept_scaling)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

