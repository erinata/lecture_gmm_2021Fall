import pandas
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pyplot

from sklearn.metrics import silhouette_score

data = pandas.read_csv("dataset.csv")

print(data)

data = data.rename({0: 'index', 'x1': 'data1', 'x2': 'data2'}, axis =1 )

print(data)

data = data.values

pyplot.scatter(data[:,1], data[:,2])
pyplot.savefig("scatter.png")
pyplot.close()

data = data[:,1:3]

print(data)

def run_kmeans(n, data):
	kmeans_machine = KMeans(n_clusters=n)
	kmean_results = kmeans_machine.fit_predict(data)
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, kmeans_machine.labels_, metric = 'euclidean')
	pyplot.scatter(data[:,0], data[:,1], c=kmean_results)
	pyplot.savefig("scatter_kmeans_" + str(n) + ".png")
	pyplot.close()
	return silhouette

def run_gmm(n, data):
	gmm_machine = GaussianMixture(n_components=n)
	gmm_results = gmm_machine.fit_predict(data)
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, gmm_results, metric = 'euclidean')
	pyplot.scatter(data[:,0], data[:,1], c=gmm_results)
	pyplot.savefig("scatter_gmm_" + str(n) + ".png")
	pyplot.close()
	return silhouette


kmeans_silhouette = run_kmeans(3, data)
gmm_silhouette = run_gmm(3, data)

print(kmeans_silhouette)
print(gmm_silhouette)




