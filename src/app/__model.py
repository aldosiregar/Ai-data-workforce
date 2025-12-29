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
    
class ModelGeneration:
    @staticmethod
    def model_generation(
        x=[], 
        preprocessing_implementation=lambda x:x,
        model_type="kmeans",
        n_cluster=3, min_sample=5):
        result = None

        match model_type:
            case "kmeans":
                result =  [KmeansModel(
                        df=preprocessing_implementation(i), 
                        n_cluster=n_cluster) for i in x]
            case "dbscan":
                result = [DbScanModel(
                        df=preprocessing_implementation(i), 
                        min_sample=min_sample) for i in x]
                
        return result