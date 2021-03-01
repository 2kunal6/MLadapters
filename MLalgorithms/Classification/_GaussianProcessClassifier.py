
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from MLalgorithms._Classification import Classification


class GaussianProcessClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
		self.warm_start = warm_start
		self.n_jobs = n_jobs
		self.optimizer = optimizer
		self.copy_X_train = copy_X_train
		self.kernel = kernel
		self.multi_class = multi_class
		self.max_iter_predict = max_iter_predict
		self.random_state = random_state
		self.n_restarts_optimizer = n_restarts_optimizer
		self.model = GPC(optimizer = self.optimizer,
			warm_start = self.warm_start,
			copy_X_train = self.copy_X_train,
			max_iter_predict = self.max_iter_predict,
			n_jobs = self.n_jobs,
			n_restarts_optimizer = self.n_restarts_optimizer,
			multi_class = self.multi_class,
			random_state = self.random_state,
			kernel = self.kernel)

