from MLalgorithms.Regression import Regression



from sklearn.linear_model import RidgeRegression


class RidgeRegression(Regression):
    
    def __init__(self, fit_intercept, normalize, n_jobs, copy_X):
        Regression.__init__(self, fit_intercept, normalize, n_jobs, copy_X)
		self.model = RidgeRegression(n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			normalize = self.normalize)
    
