
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.alphas = alphas
		self.eps = eps
		self.normalize = normalize
		self.verbose = verbose
		self.l1_ratio = l1_ratio
		self.n_alphas = n_alphas
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.cv = cv
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.max_iter = max_iter
		self.model = MTENCV(copy_X = self.copy_X,
			normalize = self.normalize,
			cv = self.cv,
			verbose = self.verbose,
			eps = self.eps,
			l1_ratio = self.l1_ratio,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			tol = self.tol,
			random_state = self.random_state,
			alphas = self.alphas,
			n_jobs = self.n_jobs,
			n_alphas = self.n_alphas)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

