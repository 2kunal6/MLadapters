





class LassoTempReg(LassoRegression):
    
    def __init__(self, copy_X, n_jobs, normalize, fit_intercept, max_iter, alpha):
        self.alpha = alpha
		LassoRegression.__init__(self, copy_X, n_jobs, normalize, fit_intercept, max_iter)

    
