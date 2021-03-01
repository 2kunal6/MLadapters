
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.dual = dual
		self.scoring = scoring
		self.Cs = Cs
		self.refit = refit
		self.l1_ratios = l1_ratios
		self.multi_class = multi_class
		self.verbose = verbose
		self.solver = solver
		self.intercept_scaling = intercept_scaling
		self.n_jobs = n_jobs
		self.penalty = penalty
		self.random_state = random_state
		self.cv = cv
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.model = LRCV(Cs = self.Cs,
			dual = self.dual,
			n_jobs = self.n_jobs,
			penalty = self.penalty,
			class_weight = self.class_weight,
			scoring = self.scoring,
			cv = self.cv,
			verbose = self.verbose,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			l1_ratios = self.l1_ratios,
			refit = self.refit,
			intercept_scaling = self.intercept_scaling,
			random_state = self.random_state,
			tol = self.tol,
			solver = self.solver,
			multi_class = self.multi_class)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

