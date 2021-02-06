


from sklearn.linear_model import Lasso


class LassoRegression(Regression):
    
    def __init__(self, n_jobs, normalize, copy_X, fit_intercept, max_iter):
        self.max_iter = max_iter
		Regression.__init__(self, n_jobs, normalize, copy_X, fit_intercept)
		self.model = Lasso(max_iter = self.max_iter,
			normalize = self.normalize,
			n_jobs = self.n_jobs,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept)
    
    
