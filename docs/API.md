
ML algorithm wrappers is a Python wrapper for algorithms defined in Scikit and Pytorch. The wrapper code is auto generated based on classes defined in ontology.

The adapters are written for Classification, Clustering, Regression and Metrics from the various models in scikit-learn.

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
| RadiusNeighborsClassifier | from sklearn.neighbors import RadiusNeighborsClassifier     |(radius=1.0, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None, n_jobs=None, **kwargs) |fit(X, y), predict(X)|
| RidgeClassifier | from sklearn.neighbors import RadiusNeighborsClassifier     |(alpha=1.0, *, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None) |fit(X, y[, sample_weight]), predict(X)|
| SGDClassifier | from sklearn.linear_model import SGDClassifier     |(loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False) |fit(X, y, coef_init=None, intercept_init=None, sample_weight=None), predict(X)|
| SVC | from sklearn.svm import SVC     |(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None) |fit(X, y[, sample_weight]), predict(X)|

# Clustering

|Algorithms|imports| parameters|methods|
| :-------------|:-------------|:-------------| :-------------|
| AffinityPropagation  |  from sklearn.cluster import AffinityPropagation  | (*, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn') | fit(X[, y]), predict(X) |
| AgglomerativeClustering     | from sklearn.cluster import AgglomerativeClustering     |   (n_clusters=2, *, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False) |fit(X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated')|
| Birch | from sklearn.cluster import Birch    |(*, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True) |fit(X[, y]), predict(X)|
| DBSCAN | from sklearn.cluster import DBSCAN     |(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None) |fit(X[, y, sample_weight])|
| GaussianMixture | from sklearn.mixture import GaussianMixture     |(n_components=1, *, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10) |fit(X[, y]), predict(X)|
| KMeans | from sklearn.cluster import KMeans     |(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto') |fit(X[, y, sample_weight]), predict(X[, sample_weight])|
| MeanShift | from sklearn.cluster import MeanShift     |(*, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300) |fit(X[, y]), predict(X)|
| OPTICS | from sklearn.cluster import OPTICS     |(*, min_samples=5, max_eps=inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None) |fit(X[, y])|
| SpectralClustering | from sklearn.cluster import SpectralClustering     |(n_clusters=8, *, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False) |fit(X[, y])|

# Regression

|Algorithms|imports| parameters|methods|
| :-------------|:-------------|:-------------| :-------------|
| AdaBoostRegressor  |  from sklearn.ensemble import AdaBoostRegressor  | (base_estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None) | fit(X, y, sample_weight=None), predict(X) |
| ARDRegression     | from sklearn.linear_model import ARDRegression    |   (*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=True, normalize=False, copy_X=True, verbose=False) |fit(X, y), predict(X[, return_std])|
| BayesianRidge | from sklearn.linear_model import BayesianRidge    |(*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False) |fit(X, y[, sample_weight]), predict(X[, return_std])|
| DecisionTreeRegressor | from sklearn.tree import DecisionTreeRegressor     |(*, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0 |fit(X, y, sample_weight=None, check_input=True, X_idx_sorted='deprecated'), predict(X, check_input=True)|
| ElasticNet | from sklearn.linear_model import ElasticNet     |(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic') |fit(X, y[, sample_weight, check_input]), predict(X)|
| GradientBoostingRegressor | from sklearn.ensemble import GradientBoostingRegressor     |(*, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0) |fit(X, y[, sample_weight, monitor]), predict(X)|
| HuberRegressor | from sklearn.linear_model import HuberRegressor     |(*, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05) |fit(X, y[, sample_weight]), predict(X)|
| LassoLars | from sklearn.linear_model import LassoLars     |(alpha=1.0, *, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None) |fit(X, y[, Xy]), predict(X)|
| LassoRegression | from sklearn.linear_model import Lasso     |(alpha=1.0, *, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic') |fit(X, y, sample_weight=None, check_input=True), predict(X)|
| LinearRegression | from sklearn.linear_model import LinearRegression     |(*, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False) |fit(X, y[, sample_weight]), predict(X)|
| LinearSVR | from sklearn.svm import LinearSVR     |(*, epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000) |fit(X, y[, sample_weight]), predict(X)|
| LogisticRegression | from sklearn.svm import LinearSVR     |(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None) |fit(X, y[, sample_weight]), predict(X)|
| LogisticRegressionCV | from sklearn.linear_model import LogisticRegressionCV    |(*, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None) |fit(X, y[, sample_weight]), predict(X)|
| MultiTaskElasticNetCV | from sklearn.linear_model import MultiTaskElasticNetCV    |(*, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic') |fit(X, y), predict(X)|
| NuSVR | from sklearn.svm import NuSVR    |(*, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=- 1) |fit(X, y[, sample_weight]), predict(X)|
| OrthogonalMatchingPursuit | from sklearn.linear_model import OrthogonalMatchingPursuit    |(*, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute='auto') |fit(X, y), predict(X)|
| RandomForestRegressor | from sklearn.ensemble import RandomForestRegressor    |(n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None) |fit(X, y[, sample_weight]), predict(X)|
| Ridge | from sklearn.linear_model import Ridge   |(alpha=1.0, *, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None) |fit(X, y[, sample_weight]), predict(X)|
| SVR | from sklearn.svm import SVR  |(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1) |fit(X, y[, sample_weight]), predict(X)|
| TweedieRegressor | from sklearn.linear_model import TweedieRegressor  |(*, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0) |fit(X, y[, sample_weight]), predict(X)|

# Metrics

|Algorithms|imports| parameters|methods|
| :-------------|:-------------|:-------------| :-------------|
| classification_report  |  from sklearn.metrics import classification_report  | (y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn') | NA |
| confusion_matrix  |  from sklearn.metrics import confusion_matrix  | (y_true, y_pred, *, labels=None, sample_weight=None, normalize=None) | NA |
| hinge_loss  |  from sklearn.metrics import hinge_loss  | (y_true, pred_decision, *, labels=None, sample_weight=None) | NA |
| jaccard_score  |  from sklearn.metrics import jaccard_score  | (y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn') | NA |
| log_loss  |  from sklearn.metrics import log_loss  | (y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None, labels=None) | NA |