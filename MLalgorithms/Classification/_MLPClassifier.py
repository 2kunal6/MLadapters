
from sklearn.neural_network import MLPClassifier as MLPC
from MLalgorithms._Classification import Classification


class MLPClassifier(Classification):
	
	def __init__(self, hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
		self.warm_start = warm_start
		self.power_t = power_t
		self.n_iter_no_change = n_iter_no_change
		self.batch_size = batch_size
		self.validation_fraction = validation_fraction
		self.random_state = random_state
		self.epsilon = epsilon
		self.learning_rate_init = learning_rate_init
		self.hidden_layer_sizes = hidden_layer_sizes
		self.alpha = alpha
		self.solver = solver
		self.shuffle = shuffle
		self.max_fun = max_fun
		self.beta_1 = beta_1
		self.momentum = momentum
		self.tol = tol
		self.nesterovs_momentum = nesterovs_momentum
		self.early_stopping = early_stopping
		self.beta_2 = beta_2
		self.activation = activation
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.model = MLPC(hidden_layer_sizes = self.hidden_layer_sizes,
			shuffle = self.shuffle,
			warm_start = self.warm_start,
			tol = self.tol,
			learning_rate_init = self.learning_rate_init,
			epsilon = self.epsilon,
			nesterovs_momentum = self.nesterovs_momentum,
			n_iter_no_change = self.n_iter_no_change,
			max_fun = self.max_fun,
			activation = self.activation,
			beta_2 = self.beta_2,
			learning_rate = self.learning_rate,
			validation_fraction = self.validation_fraction,
			alpha = self.alpha,
			batch_size = self.batch_size,
			early_stopping = self.early_stopping,
			momentum = self.momentum,
			max_iter = self.max_iter,
			beta_1 = self.beta_1,
			power_t = self.power_t,
			random_state = self.random_state,
			solver = self.solver)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

