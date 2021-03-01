
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from MLalgorithms._Classification import Classification


class GaussianProcessClassifier(Classification):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
		self.warm_start = warm_start
		self.max_iter_predict = max_iter_predict
		self.optimizer = optimizer
		self.multi_class = multi_class
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.kernel = kernel
		self.n_restarts_optimizer = n_restarts_optimizer
		self.copy_X_train = copy_X_train
		self.model = GPC(copy_X_train = self.copy_X_train,
			multi_class = self.multi_class,
			warm_start = self.warm_start,
			n_restarts_optimizer = self.n_restarts_optimizer,
			max_iter_predict = self.max_iter_predict,
			random_state = self.random_state,
			optimizer = self.optimizer,
			n_jobs = self.n_jobs,
			kernel = self.kernel)

	def predict(self, X):
		return self.model.predict(X=X)

