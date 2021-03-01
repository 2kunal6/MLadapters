
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.tol = tol
		self.max_fun = max_fun
		self.nesterovs_momentum = nesterovs_momentum
		self.epsilon = epsilon
		self.shuffle = shuffle
		self.warm_start = warm_start
		self.solver = solver
		self.validation_fraction = validation_fraction
		self.batch_size = batch_size
		self.early_stopping = early_stopping
		self.activation = activation
		self.hidden_layer_sizes = hidden_layer_sizes
		self.n_iter_no_change = n_iter_no_change
		self.alpha = alpha
		self.max_iter = max_iter
		self.momentum = momentum
		self.learning_rate = learning_rate
		self.learning_rate_init = learning_rate_init
		self.beta_1 = beta_1
		self.power_t = power_t
		self.beta_2 = beta_2
		self.random_state = random_state
		self.model = MLPC(max_fun = self.max_fun,
			random_state = self.random_state,
			hidden_layer_sizes = self.hidden_layer_sizes,
			validation_fraction = self.validation_fraction,
			momentum = self.momentum,
			nesterovs_momentum = self.nesterovs_momentum,
			tol = self.tol,
			epsilon = self.epsilon,
			beta_2 = self.beta_2,
			shuffle = self.shuffle,
			learning_rate_init = self.learning_rate_init,
			warm_start = self.warm_start,
			early_stopping = self.early_stopping,
			n_iter_no_change = self.n_iter_no_change,
			activation = self.activation,
			solver = self.solver,
			batch_size = self.batch_size,
			alpha = self.alpha,
			learning_rate = self.learning_rate,
			max_iter = self.max_iter,
			beta_1 = self.beta_1,
			power_t = self.power_t)

