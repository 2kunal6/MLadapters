
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from MLalgorithms._Classification import Classification


class GaussianProcessClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
		self.copy_X_train = copy_X_train
		self.random_state = random_state
		self.optimizer = optimizer
		self.max_iter_predict = max_iter_predict
		self.warm_start = warm_start
		self.n_jobs = n_jobs
		self.kernel = kernel
		self.multi_class = multi_class
		self.n_restarts_optimizer = n_restarts_optimizer
		self.model = GPC(optimizer = self.optimizer,
			n_jobs = self.n_jobs,
			copy_X_train = self.copy_X_train,
			random_state = self.random_state,
			max_iter_predict = self.max_iter_predict,
			multi_class = self.multi_class,
			kernel = self.kernel,
			n_restarts_optimizer = self.n_restarts_optimizer,
			warm_start = self.warm_start)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

