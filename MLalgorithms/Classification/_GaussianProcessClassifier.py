
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from MLalgorithms._Classification import Classification


class GaussianProcessClassifier(Classification):
	
	def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
		self.copy_X_train = copy_X_train
		self.multi_class = multi_class
		self.random_state = random_state
		self.kernel = kernel
		self.warm_start = warm_start
		self.optimizer = optimizer
		self.n_restarts_optimizer = n_restarts_optimizer
		self.max_iter_predict = max_iter_predict
		self.n_jobs = n_jobs
		self.model = GPC(random_state = self.random_state,
			copy_X_train = self.copy_X_train,
			multi_class = self.multi_class,
			optimizer = self.optimizer,
			n_restarts_optimizer = self.n_restarts_optimizer,
			max_iter_predict = self.max_iter_predict,
			warm_start = self.warm_start,
			kernel = self.kernel,
			n_jobs = self.n_jobs)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

