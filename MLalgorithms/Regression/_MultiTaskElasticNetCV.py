
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.n_alphas = n_alphas
		self.copy_X = copy_X
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.alphas = alphas
		self.verbose = verbose
		self.tol = tol
		self.eps = eps
		self.selection = selection
		self.cv = cv
		self.normalize = normalize
		self.fit_intercept = fit_intercept
		self.l1_ratio = l1_ratio
		self.model = MTENCV(tol = self.tol,
			eps = self.eps,
			alphas = self.alphas,
			copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			normalize = self.normalize,
			l1_ratio = self.l1_ratio,
			random_state = self.random_state,
			cv = self.cv,
			n_alphas = self.n_alphas,
			max_iter = self.max_iter,
			verbose = self.verbose,
			selection = self.selection,
			fit_intercept = self.fit_intercept)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

