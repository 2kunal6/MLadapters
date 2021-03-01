
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.early_stopping = early_stopping
		self.n_iter_no_change = n_iter_no_change
		self.nesterovs_momentum = nesterovs_momentum
		self.beta_2 = beta_2
		self.beta_1 = beta_1
		self.shuffle = shuffle
		self.activation = activation
		self.tol = tol
		self.random_state = random_state
		self.learning_rate_init = learning_rate_init
		self.warm_start = warm_start
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.momentum = momentum
		self.validation_fraction = validation_fraction
		self.batch_size = batch_size
		self.solver = solver
		self.max_fun = max_fun
		self.hidden_layer_sizes = hidden_layer_sizes
		self.power_t = power_t
		self.model = MLPC(tol = self.tol,
			max_fun = self.max_fun,
			nesterovs_momentum = self.nesterovs_momentum,
			learning_rate = self.learning_rate,
			shuffle = self.shuffle,
			n_iter_no_change = self.n_iter_no_change,
			validation_fraction = self.validation_fraction,
			random_state = self.random_state,
			warm_start = self.warm_start,
			activation = self.activation,
			momentum = self.momentum,
			hidden_layer_sizes = self.hidden_layer_sizes,
			alpha = self.alpha,
			beta_2 = self.beta_2,
			epsilon = self.epsilon,
			solver = self.solver,
			learning_rate_init = self.learning_rate_init,
			early_stopping = self.early_stopping,
			beta_1 = self.beta_1,
			power_t = self.power_t,
			batch_size = self.batch_size,
			max_iter = self.max_iter)

