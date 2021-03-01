
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.learning_rate = learning_rate
		self.max_fun = max_fun
		self.n_iter_no_change = n_iter_no_change
		self.validation_fraction = validation_fraction
		self.early_stopping = early_stopping
		self.beta_2 = beta_2
		self.max_iter = max_iter
		self.solver = solver
		self.beta_1 = beta_1
		self.tol = tol
		self.warm_start = warm_start
		self.learning_rate_init = learning_rate_init
		self.shuffle = shuffle
		self.nesterovs_momentum = nesterovs_momentum
		self.power_t = power_t
		self.random_state = random_state
		self.alpha = alpha
		self.momentum = momentum
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation = activation
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.model = MLPC(n_iter_no_change = self.n_iter_no_change,
			warm_start = self.warm_start,
			alpha = self.alpha,
			epsilon = self.epsilon,
			early_stopping = self.early_stopping,
			random_state = self.random_state,
			beta_1 = self.beta_1,
			shuffle = self.shuffle,
			max_fun = self.max_fun,
			learning_rate = self.learning_rate,
			beta_2 = self.beta_2,
			tol = self.tol,
			activation = self.activation,
			power_t = self.power_t,
			hidden_layer_sizes = self.hidden_layer_sizes,
			momentum = self.momentum,
			solver = self.solver,
			batch_size = self.batch_size,
			learning_rate_init = self.learning_rate_init,
			max_iter = self.max_iter,
			validation_fraction = self.validation_fraction,
			nesterovs_momentum = self.nesterovs_momentum)

