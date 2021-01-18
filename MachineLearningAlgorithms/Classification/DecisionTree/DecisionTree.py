

class DecisionTree(Classification):


'''['DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.\n\nAs with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, of shape (n_samples, n_features) holding the training samples, and an array Y of integer values, shape (n_samples,), holding the class labels for the training samples.']'''

	def apply(X,check_input):
	'''['Return the index of the leaf that each sample is predicted as.']'''
		pass

	def cost_complexity_pruning_path(X,y,sample_weight):
	'''['Compute the pruning path during Minimal Cost-Complexity Pruning.']'''
		pass

	def decision_path(X,check_input):
	'''['Return the decision path in the tree.']'''
		pass

	def fit(X,y,sample_weight,check_input,X_idx_sorted):
	'''['Build a decision tree classifier from the training set (X, y).']'''
		self._model.fit(X, y)

	def get_depth():
	'''['Return the depth of the decision tree.']'''
		pass

	def get_n_leaves():
	'''['Return the number of leaves of the decision tree.']'''
		pass

	def get_params(deep):
	'''['Get parameters for this estimator.']'''
		pass

	def predict(X,check_input):
	'''['Predict class or regression value for X.']'''
		self._model.predict(X)

	def predict_log_proba():
	'''['Predict class log-probabilities of the input samples X.']'''
		pass