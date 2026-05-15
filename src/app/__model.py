from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from pandas import DataFrame
from numpy import array

class KmeansModel:
    def __init__(self, df=DataFrame([]), n_cluster=3):
        self.df = df
        self.model = KMeans(n_clusters=n_cluster)
        self.model.fit(df)

    def predict(self, df=DataFrame([])) -> array:
        return self.model.predict(df)
    
class NearestMatch:
    def __init__(self, df=DataFrame([]), n_neighbors=5):
        self.df = df
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.model.fit(df)
    
class ModelGeneration:
    @staticmethod
    def model_generation(
        x=[], 
        preprocessing_implementation=lambda x:x,
        model_type="kmeans",
        n_cluster=3, n_neighbors=5):
        result = None

        match model_type:
            case "kmeans":
                result =  [KmeansModel(
                        df=preprocessing_implementation(i), 
                        n_cluster=n_cluster) for i in x]
            case "NearestNeighbors":
                result = [NearestMatch(
                        df=preprocessing_implementation(i), 
                        n_neighbors=n_neighbors) for i in x]
                
        return result