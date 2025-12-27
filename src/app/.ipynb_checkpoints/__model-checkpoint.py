from sklearn.cluster import KMeans, DBSCAN
from pandas import DataFrame
from numpy import array

class KmeansModel:
    def __init__(self, df=DataFrame([]), n_cluster=3):
        self.df = df
        self.model = KMeans(n_clusters=n_cluster)
        self.model.fit(df)

    def predict(self, df=DataFrame([])) -> array:
        return self.model.predict(df)
    
class DbScanModel:
    def __init__(self, df=DataFrame([]), eps=0.5, min_sample=5):
        self.df = df
        self.model = DBSCAN(eps=eps, min_samples=min_sample)

    def predict(self, df=DataFrame([]))-> array:
        return self.model.fit_predict(df)