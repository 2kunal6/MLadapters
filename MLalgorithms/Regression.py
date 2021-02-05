





class Regression(MLalgorithms):
    
    def predict(self, X):
        return self.model.predict(X=X)
    
    
    def fit(self, y, sample_weight, X):
        return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)
    
    
    def __init__(self, copy_X, n_jobs = None, fit_intercept, normalize):
        self.copy_X = copy_X
		self.n_jobs = n_jobs
		self.fit_intercept = fit_intercept
		self.normalize = normalize
    
    
