import MLalgorithms






class Regression(MLalgorithms):
    
    def __init__(self, fit_intercept, normalize, n_jobs = None, copy_X):
        self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.n_jobs = n_jobs
		self.copy_X = copy_X

    
    def predict(self, X):
        return self.model.predict(X=X)

    
    def fit(self, sample_weight, y, X):
        return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

    
