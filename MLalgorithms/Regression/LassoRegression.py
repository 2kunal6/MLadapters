


from sklearn.linear_model import Lasso


class LassoRegression(Regression):
    
    def __init__(self, n_jobs, normalize, copy_X, max_iter, fit_intercept, max_iter, max_iter):
        self.max_iter = max_iter
		Regression.__init__(self, n_jobs, normalize, copy_X, max_iter, fit_intercept, max_iter)
		self.model = Lasso(n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			normalize = self.normalize,
			max_iter = self.max_iter)
    
    
