
import numpy as np
from sklearn.cluster import MeanShift as MSClustering
from MLalgorithms._Clustering import Clustering


class MeanShift(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300):
		self.max_iter = max_iter
		self.seeds = seeds
		self.bandwidth = bandwidth
		self.bin_seeding = bin_seeding
		self.min_bin_freq = min_bin_freq
		self.n_jobs = n_jobs
		self.cluster_all = cluster_all
		self.model = MSClustering(bandwidth = self.bandwidth,
			seeds = self.seeds,
			bin_seeding = self.bin_seeding,
			min_bin_freq = self.min_bin_freq,
			n_jobs = self.n_jobs,
			max_iter = self.max_iter,
			cluster_all = self.cluster_all)

	def predict(self, X):
		return self.model.predict(X=X)

