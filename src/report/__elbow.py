from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from scipy.optimize import curve_fit
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

    """
    @staticmethod
    def derivation_report(x=DataFrame([])):
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

        weight = Regression.PolynomialModel(K, distortions, degree=3)

        change = np.array(
            [
                (
                    (weight[3] * (i ** 3)) + (weight[2] * (i ** 2)) + (weight[1] * i) + (weight[0]) 
                ) for i in K
            ]
        )

        plot(K, change, "bx-")
        xlabel('Number of Clusters (k)')
        ylabel('Change')
        title('The Elbow Method distortion change')
        show()
        """
    
class CurveFitting:
    @staticmethod
    def PolynomialModel(x=np.array([]), y=np.array([]), degree=2):
        return np.polyfit(x=x, y=y, deg=degree)