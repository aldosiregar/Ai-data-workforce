from sklearn.cluster import KMeans
from pandas import DataFrame
from numpy import array

class KmeansModel:
    def __init__(self, df=DataFrame([]), n_cluster=3):
        self.df = df
        self.model = KMeans(n_clusters=n_cluster)
        self.model.fit(df)

    def predict(self, df=DataFrame([])) -> array:
        return self.model.predict(df)