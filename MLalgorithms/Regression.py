





class Regression(MLalgorithms):
    
    def fit(self, X, sample_weight, y):
        None

    
    def predict(self, X):
        None

    
    def __init__(self, copy_X, n_jobs = None, normalize, fit_intercept):
        self.copy_X = copy_X
		self.n_jobs = n_jobs
		self.normalize = normalize
		self.fit_intercept = fit_intercept

    
