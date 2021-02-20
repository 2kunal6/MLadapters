from MLalgorithms.Regression import Regression



from sklearn.linear_model import Lasso


class LassoRegression(Regression):
    
    def __init__(self, copy_X, fit_intercept, normalize, n_jobs, max_iter):
        self.max_iter = max_iter
		Regression.__init__(self, copy_X, fit_intercept, normalize, n_jobs)
		self.model = Lasso(fit_intercept = self.fit_intercept,
			n_jobs = self.n_jobs,
			copy_X = self.copy_X,
			max_iter = self.max_iter,
			normalize = self.normalize)

    
