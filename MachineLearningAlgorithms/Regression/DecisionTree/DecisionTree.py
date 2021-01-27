

from MachineLearningAlgorithms.Regression.Regression import Regression

from sklearn.tree import DecisionTreeRegressor

'''Decision trees can be applied to regression problems, using the DecisionTreeRegressor class.

As in the classification setting, the fit method will take as argument arrays X and y, only that in this case y is expected to have floating point values instead of integer values.'''

class DecisionTree(Regression):


	def __init__(self, ccp_alpha,class_weight,criterion,max_depth,max_features,max_leaf_nodes,min_impurity_decrease,min_impurity_split,min_samples_leaf,min_samples_split,min_weight_fraction_leaf,random_state,splitter):
		self._model = DecisionTreeRegressor(ccp_alpha,class_weight,criterion,max_depth,max_features,max_leaf_nodes,min_impurity_decrease,min_impurity_split,min_samples_leaf,min_samples_split,min_weight_fraction_leaf,random_state,splitter)

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