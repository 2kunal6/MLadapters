





class Regression(MLalgorithms):
    
    def __init__(self, n_jobs = None, normalize, copy_X, fit_intercept):
        self.n_jobs = n_jobs
		self.normalize = normalize
		self.copy_X = copy_X
		self.fit_intercept = fit_intercept
    
    
    def fit(self, y, X, sample_weight):
        return self.model.fit(y=y,
			X=X,
			sample_weight=sample_weight)
    
    
    def predict(self, X):
        return self.model.predict(X=X)
    
    
