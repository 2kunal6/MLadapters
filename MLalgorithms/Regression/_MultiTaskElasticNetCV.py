
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.verbose = verbose
		self.l1_ratio = l1_ratio
		self.tol = tol
		self.cv = cv
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.alphas = alphas
		self.fit_intercept = fit_intercept
		self.n_alphas = n_alphas
		self.copy_X = copy_X
		self.normalize = normalize
		self.eps = eps
		self.random_state = random_state
		self.selection = selection
		self.model = MTENCV(l1_ratio = self.l1_ratio,
			selection = self.selection,
			normalize = self.normalize,
			verbose = self.verbose,
			n_alphas = self.n_alphas,
			cv = self.cv,
			max_iter = self.max_iter,
			alphas = self.alphas,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state,
			copy_X = self.copy_X,
			eps = self.eps,
			n_jobs = self.n_jobs)

