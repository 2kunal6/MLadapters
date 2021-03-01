
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.early_stopping = early_stopping
		self.shuffle = shuffle
		self.learning_rate_init = learning_rate_init
		self.random_state = random_state
		self.power_t = power_t
		self.beta_2 = beta_2
		self.beta_1 = beta_1
		self.epsilon = epsilon
		self.validation_fraction = validation_fraction
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.momentum = momentum
		self.learning_rate = learning_rate
		self.tol = tol
		self.solver = solver
		self.max_fun = max_fun
		self.alpha = alpha
		self.nesterovs_momentum = nesterovs_momentum
		self.hidden_layer_sizes = hidden_layer_sizes
		self.n_iter_no_change = n_iter_no_change
		self.activation = activation
		self.warm_start = warm_start
		self.model = MLPC(random_state = self.random_state,
			power_t = self.power_t,
			momentum = self.momentum,
			learning_rate = self.learning_rate,
			solver = self.solver,
			validation_fraction = self.validation_fraction,
			n_iter_no_change = self.n_iter_no_change,
			beta_2 = self.beta_2,
			epsilon = self.epsilon,
			hidden_layer_sizes = self.hidden_layer_sizes,
			alpha = self.alpha,
			nesterovs_momentum = self.nesterovs_momentum,
			max_fun = self.max_fun,
			tol = self.tol,
			learning_rate_init = self.learning_rate_init,
			batch_size = self.batch_size,
			early_stopping = self.early_stopping,
			max_iter = self.max_iter,
			beta_1 = self.beta_1,
			warm_start = self.warm_start,
			activation = self.activation,
			shuffle = self.shuffle)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

