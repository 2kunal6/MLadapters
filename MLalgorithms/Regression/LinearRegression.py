


from sklearn.linear_model import LinearRegression


class LinearRegression(Regression):
    
    def __init__(self, n_jobs, normalize, copy_X, fit_intercept):
        Regression.__init__(self, n_jobs, normalize, copy_X, fit_intercept)
		self.model = LinearRegression(copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize)
    
