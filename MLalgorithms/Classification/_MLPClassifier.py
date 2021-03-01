
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.activation = activation
		self.max_fun = max_fun
		self.beta_2 = beta_2
		self.solver = solver
		self.learning_rate = learning_rate
		self.shuffle = shuffle
		self.random_state = random_state
		self.learning_rate_init = learning_rate_init
		self.alpha = alpha
		self.warm_start = warm_start
		self.beta_1 = beta_1
		self.nesterovs_momentum = nesterovs_momentum
		self.power_t = power_t
		self.early_stopping = early_stopping
		self.batch_size = batch_size
		self.hidden_layer_sizes = hidden_layer_sizes
		self.n_iter_no_change = n_iter_no_change
		self.max_iter = max_iter
		self.epsilon = epsilon
		self.momentum = momentum
		self.validation_fraction = validation_fraction
		self.tol = tol
		self.model = MLPC(nesterovs_momentum = self.nesterovs_momentum,
			alpha = self.alpha,
			solver = self.solver,
			hidden_layer_sizes = self.hidden_layer_sizes,
			n_iter_no_change = self.n_iter_no_change,
			validation_fraction = self.validation_fraction,
			max_fun = self.max_fun,
			learning_rate_init = self.learning_rate_init,
			beta_2 = self.beta_2,
			early_stopping = self.early_stopping,
			epsilon = self.epsilon,
			activation = self.activation,
			batch_size = self.batch_size,
			random_state = self.random_state,
			tol = self.tol,
			shuffle = self.shuffle,
			momentum = self.momentum,
			beta_1 = self.beta_1,
			max_iter = self.max_iter,
			warm_start = self.warm_start,
			power_t = self.power_t,
			learning_rate = self.learning_rate)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

