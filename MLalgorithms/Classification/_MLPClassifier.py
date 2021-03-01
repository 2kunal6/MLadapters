
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.max_fun = max_fun
		self.tol = tol
		self.random_state = random_state
		self.beta_2 = beta_2
		self.warm_start = warm_start
		self.power_t = power_t
		self.learning_rate = learning_rate
		self.early_stopping = early_stopping
		self.hidden_layer_sizes = hidden_layer_sizes
		self.solver = solver
		self.shuffle = shuffle
		self.n_iter_no_change = n_iter_no_change
		self.batch_size = batch_size
		self.max_iter = max_iter
		self.validation_fraction = validation_fraction
		self.nesterovs_momentum = nesterovs_momentum
		self.momentum = momentum
		self.activation = activation
		self.beta_1 = beta_1
		self.alpha = alpha
		self.learning_rate_init = learning_rate_init
		self.epsilon = epsilon
		self.model = MLPC(random_state = self.random_state,
			power_t = self.power_t,
			alpha = self.alpha,
			epsilon = self.epsilon,
			batch_size = self.batch_size,
			tol = self.tol,
			early_stopping = self.early_stopping,
			momentum = self.momentum,
			beta_1 = self.beta_1,
			solver = self.solver,
			warm_start = self.warm_start,
			shuffle = self.shuffle,
			n_iter_no_change = self.n_iter_no_change,
			beta_2 = self.beta_2,
			max_fun = self.max_fun,
			nesterovs_momentum = self.nesterovs_momentum,
			validation_fraction = self.validation_fraction,
			learning_rate_init = self.learning_rate_init,
			learning_rate = self.learning_rate,
			hidden_layer_sizes = self.hidden_layer_sizes,
			max_iter = self.max_iter,
			activation = self.activation)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

