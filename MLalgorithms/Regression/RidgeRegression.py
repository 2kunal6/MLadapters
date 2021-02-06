


from sklearn.linear_model import RidgeRegression


class RidgeRegression(Regression):
    
    def __init__(self, n_jobs, normalize, copy_X, fit_intercept):
        Regression.__init__(self, n_jobs, normalize, copy_X, fit_intercept)
		self.model = RidgeRegression(copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize)
    
