
ML algorithm wrappers is a Python wrapper for algorithms defined in Scikit and Pytorch. The wrapper code is auto generated based on classes defined in ontology.

The adapters are written for Classification, Clustering, Regression and Matrices from the various models in scikit-learn.

The following sections give all the information about the various algorithm implemented in the project.

# Classification

|Algorithms|imports| parameters|methods|
| :-------------|:-------------|:-------------| :-------------|
| AdaBoostClassifier  |  from sklearn.ensemble import AdaBoostClassifier  | (base_estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None) | fit(X, y[, sample_weight]), predict(X) |
| DecisionTreeClassifier     | from sklearn.tree import DecisionTreeClassifier     |   (*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0) |fit(X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'), predict(X, check_input=True)|
| GaussianProcessClassifier | from sklearn.gaussian_process import GaussianProcessClassifier     |(kernel=None, *, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None) |fit(X, y), predict(X)|
| KNeighborsClassifier | from sklearn.neighbors import KNeighborsClassifier     |(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs) |fit(X, y), predict(X)|
| MLPClassifier | from sklearn.neural_network import MLPClassifier     |(hidden_layer_sizes=100, activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000) |fit(X, y), predict(X)|
| NearestCentroid | from sklearn.neighbors import NearestCentroid     |(metric='euclidean', *, shrink_threshold=None) |fit(X, y), predict(X)|
