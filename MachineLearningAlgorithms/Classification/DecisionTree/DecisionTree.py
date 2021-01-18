

class DecisionTree(Classification):


'''['DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.\n\nAs with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, of shape (n_samples, n_features) holding the training samples, and an array Y of integer values, shape (n_samples,), holding the class labels for the training samples.']'''

	def __init__(self, class_weight:dict = None,criterion:None,max_depth:int = None,min_samples_split:float = 2,splitter:str = ”best”):
		self.class_weight = class_weight
		self.criterion = criterion
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.splitter = splitter


	def fit(X,y,sample_weight,check_input):
	'''Build a decision tree classifier from the training set (X, y).'''
		self._model.fit(X, y)

	def get_depth():
	'''Return the depth of the decision tree.'''
		pass

	def get_n_leaves():
	'''Return the number of leaves of the decision tree.'''
		pass

	def apply(X,check_input):
	'''Return the index of the leaf that each sample is predicted as.

Parameters: 
X{array-like, sparse matrix} of shape (n_samples, n_features)
The input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csr_matrix.

check_inputbool, default=True
Allow to bypass several input checking. Don’t use this parameter unless you know what you do.


Returns:
X_leavesarray-like of shape (n_samples,)
For each datapoint x in X, return the index of the leaf x ends up in. Leaves are numbered within [0; self.tree_.node_count), possibly with gaps in the numbering.'''
		pass

	def predict(X,check_input):
	'''Predict class or regression value for X.

Parameters:
X{array-like, sparse matrix} of shape (n_samples, n_features)
The input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csr_matrix.


Returns:
probandarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1
The class log-probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.'''
		self._model.predict(X)