
from sklearn.cluster import MeanShift as MSClustering
from MLalgorithms._Clustering import Clustering


class MeanShift(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
		self.cluster_all = cluster_all
		self.bin_seeding = bin_seeding
		self.bandwidth = bandwidth
		self.seeds = seeds
		self.max_iter = max_iter
		self.n_jobs = n_jobs
		self.min_bin_freq = min_bin_freq
		self.model = MSClustering(bandwidth = self.bandwidth,
			bin_seeding = self.bin_seeding,
			max_iter = self.max_iter,
			seeds = self.seeds,
			cluster_all = self.cluster_all,
			n_jobs = self.n_jobs,
			min_bin_freq = self.min_bin_freq)

