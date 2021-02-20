from MLalgorithms.Regression import Regression



from sklearn.linear_model import LinearRegression


class LinearRegression(Regression):
    
    def __init__(self, copy_X, fit_intercept, normalize, n_jobs):
        Regression.__init__(self, copy_X, fit_intercept, normalize, n_jobs)
		self.model = LinearRegression(fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			normalize = self.normalize)
    
