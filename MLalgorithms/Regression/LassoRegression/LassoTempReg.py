





class LassoTempReg(LassoRegression):
    
    def __init__(self, n_jobs, normalize, copy_X, fit_intercept, max_iter, alpha):
        self.alpha = alpha
		LassoRegression.__init__(self, n_jobs, normalize, copy_X, fit_intercept, max_iter)
    
    
