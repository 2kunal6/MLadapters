from MLalgorithms.Regression.LassoRegression import LassoRegression






class LassoTempReg(LassoRegression):
    
    def __init__(self, fit_intercept, normalize, n_jobs, copy_X, max_iter, alpha):
        self.alpha = alpha
		LassoRegression.__init__(self, fit_intercept, normalize, n_jobs, copy_X, max_iter)

    
