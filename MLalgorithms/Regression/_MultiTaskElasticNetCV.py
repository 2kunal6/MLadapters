
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.eps = eps
		self.copy_X = copy_X
		self.cv = cv
		self.l1_ratio = l1_ratio
		self.normalize = normalize
		self.max_iter = max_iter
		self.random_state = random_state
		self.verbose = verbose
		self.n_jobs = n_jobs
		self.tol = tol
		self.alphas = alphas
		self.selection = selection
		self.n_alphas = n_alphas
		self.fit_intercept = fit_intercept
		self.model = MTENCV(max_iter = self.max_iter,
			alphas = self.alphas,
			eps = self.eps,
			normalize = self.normalize,
			copy_X = self.copy_X,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			n_alphas = self.n_alphas,
			cv = self.cv,
			l1_ratio = self.l1_ratio,
			random_state = self.random_state,
			tol = self.tol,
			n_jobs = self.n_jobs,
			verbose = self.verbose)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

