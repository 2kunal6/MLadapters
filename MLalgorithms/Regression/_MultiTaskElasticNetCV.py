
from sklearn.linear_model import MultiTaskElasticNetCV as MTENCV
from MLalgorithms._Regression import Regression


class MultiTaskElasticNetCV(Regression):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=None, alphas=None, fit_intercept=True, normalize=False, max_iter=100, tol=0.0001, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
		self.alphas = alphas
		self.n_alphas = n_alphas
		self.verbose = verbose
		self.max_iter = max_iter
		self.eps = eps
		self.tol = tol
		self.random_state = random_state
		self.selection = selection
		self.n_jobs = n_jobs
		self.l1_ratio = l1_ratio
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = MTENCV(alphas = self.alphas,
			max_iter = self.max_iter,
			l1_ratio = self.l1_ratio,
			n_alphas = self.n_alphas,
			copy_X = self.copy_X,
			normalize = self.normalize,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			n_jobs = self.n_jobs,
			verbose = self.verbose,
			selection = self.selection,
			eps = self.eps,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

