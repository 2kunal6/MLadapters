
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.beta_1 = beta_1
		self.n_iter_no_change = n_iter_no_change
		self.momentum = momentum
		self.max_fun = max_fun
		self.beta_2 = beta_2
		self.max_iter = max_iter
		self.solver = solver
		self.hidden_layer_sizes = hidden_layer_sizes
		self.validation_fraction = validation_fraction
		self.random_state = random_state
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.nesterovs_momentum = nesterovs_momentum
		self.warm_start = warm_start
		self.learning_rate = learning_rate
		self.alpha = alpha
		self.tol = tol
		self.learning_rate_init = learning_rate_init
		self.power_t = power_t
		self.activation = activation
		self.epsilon = epsilon
		self.early_stopping = early_stopping
		self.model = MLPC(learning_rate_init = self.learning_rate_init,
			validation_fraction = self.validation_fraction,
			max_iter = self.max_iter,
			momentum = self.momentum,
			warm_start = self.warm_start,
			beta_1 = self.beta_1,
			beta_2 = self.beta_2,
			batch_size = self.batch_size,
			shuffle = self.shuffle,
			epsilon = self.epsilon,
			solver = self.solver,
			n_iter_no_change = self.n_iter_no_change,
			alpha = self.alpha,
			hidden_layer_sizes = self.hidden_layer_sizes,
			random_state = self.random_state,
			learning_rate = self.learning_rate,
			early_stopping = self.early_stopping,
			max_fun = self.max_fun,
			power_t = self.power_t,
			nesterovs_momentum = self.nesterovs_momentum,
			tol = self.tol,
			activation = self.activation)

	def predict(self, X):
		return self.model.predict(X=X)

