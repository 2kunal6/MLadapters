





class LassoTempReg(LassoRegression):
    
    def __init__(self, copy_X, n_jobs, fit_intercept, normalize, max_iter, alpha):
        self.alpha = alpha
		LassoRegression.__init__(self, copy_X, n_jobs, fit_intercept, normalize, max_iter)
    
    
