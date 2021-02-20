import MLalgorithms






class Regression(MLalgorithms):
    
    def predict(self, X):
        return self.model.predict(X=X)

    
    def __init__(self, copy_X, fit_intercept, normalize, n_jobs = None):
        self.copy_X = copy_X
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.n_jobs = n_jobs

    
    def fit(self, X, sample_weight, y):
        return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

    
