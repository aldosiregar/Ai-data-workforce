from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, show, title
from pandas import DataFrame

class ElbowMethod:
    @staticmethod
    def report(x=DataFrame([])):
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = np.arange(1, 10)

        X = x

        for k in K:
            kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
            
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])
            
            inertias.append(kmeanModel.inertia_)
            
            mapping1[k] = distortions[-1]
            mapping2[k] = inertias[-1]

        plot(K, distortions, 'bx-')
        xlabel('Number of Clusters (k)')
        ylabel('Distortion')
        title('The Elbow Method using Distortion')
        show()