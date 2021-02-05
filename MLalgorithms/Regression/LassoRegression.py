


from sklearn.linear_model import Lasso


class LassoRegression(Regression):
    
    def __init__(self, copy_X, n_jobs, fit_intercept, normalize, max_iter):
        self.max_iter = max_iter
		Regression.__init__(self, copy_X, n_jobs, fit_intercept, normalize)
		self.model = Lasso(max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			n_jobs = self.n_jobs,
			copy_X = self.copy_X)
    
    
