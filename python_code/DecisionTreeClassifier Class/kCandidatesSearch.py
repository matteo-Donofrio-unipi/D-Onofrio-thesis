from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def runKMeans():


    X = load_iris().data
    df=pd.DataFrame(X,columns=['att1','att2','att3','att4'])
    print(df)
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(X)

    centroids = kmeans.cluster_centers_


    medoids = list()
    for label in np.unique(kmeans.labels_):
        X_cluster = X[kmeans.labels_ == label]
        dist = pdist(X_cluster)
        index = np.argmin(np.sum(squareform(dist), axis=0))
        medoids.append(X_cluster[index])
    medoids = np.array(medoids)

    print('\n')
    print(medoids)

