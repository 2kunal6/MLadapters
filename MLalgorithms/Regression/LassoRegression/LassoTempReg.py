





class LassoTempReg(LassoRegression):
    
    def __init__(self, n_jobs, normalize, copy_X, max_iter, fit_intercept, max_iter, alpha = "Hi"):
        self.alpha = alpha
		LassoRegression.__init__(self, n_jobs, normalize, copy_X, max_iter, fit_intercept, max_iter)
    
    
