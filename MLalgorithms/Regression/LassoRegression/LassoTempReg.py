from MLalgorithms.Regression.LassoRegression import LassoRegression






class LassoTempReg(LassoRegression):
    
    def __init__(self, copy_X, fit_intercept, normalize, n_jobs, max_iter, alpha):
        self.alpha = alpha
		LassoRegression.__init__(self, copy_X, fit_intercept, normalize, n_jobs, max_iter)

    
