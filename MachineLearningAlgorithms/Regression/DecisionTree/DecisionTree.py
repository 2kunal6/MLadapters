

from MachineLearningAlgorithms.Regression.Regression import Regression

from sklearn.tree import DecisionTreeRegressor

'''Decision trees can be applied to regression problems, using the DecisionTreeRegressor class.

As in the classification setting, the fit method will take as argument arrays X and y, only that in this case y is expected to have floating point values instead of integer values.'''

class DecisionTree(Regression):


	def fit(self, X=None,y=None,sample_weight=None,check_input=None):
		'''Build a decision tree classifier from the training set (X, y).'''
		return self._model.fit(X, y)

	def get_depth(self, ):
		'''Return the depth of the decision tree.'''
		pass

	def get_n_leaves(self, ):
		'''Return the number of leaves of the decision tree.'''
		pass

	def __init__(self, criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,class_weight=None,ccp_alpha=0.0):
		'''None'''
		self._model = DecisionTreeRegressor(criterion,splitter,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_features,random_state,max_leaf_nodes,min_impurity_decrease,min_impurity_split,class_weight,ccp_alpha)


	def apply(self, X=None,check_input=None):
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

	def predict(self, X=None,check_input=None):
		'''Predict class or regression value for X.

Parameters:
X{array-like, sparse matrix} of shape (n_samples, n_features)
The input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csr_matrix.


Returns:
probandarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1
The class log-probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.'''
		self._model.predict(X)