
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.tol = tol
		self.n_alphas = n_alphas
		self.n_jobs = n_jobs
		self.selection = selection
		self.normalize = normalize
		self.l1_ratio = l1_ratio
		self.verbose = verbose
		self.random_state = random_state
		self.cv = cv
		self.eps = eps
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.alphas = alphas
		self.copy_X = copy_X
		self.model = MTENCV(max_iter = self.max_iter,
			n_alphas = self.n_alphas,
			normalize = self.normalize,
			l1_ratio = self.l1_ratio,
			n_jobs = self.n_jobs,
			selection = self.selection,
			eps = self.eps,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			alphas = self.alphas,
			copy_X = self.copy_X,
			verbose = self.verbose,
			cv = self.cv,
			random_state = self.random_state)

