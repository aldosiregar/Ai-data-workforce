from sklearn.metrics import silhouette_score, davies_bouldin_score
from pandas import DataFrame
from sklearn.cluster import KMeans, DBSCAN

class SilhoutteScore:
    def __init__(self, x=DataFrame([]), model=KMeans|DBSCAN):
        self.x = x
        self.model = model

    @staticmethod 
    def batch_scoring(x=[DataFrame([])], model=[KMeans|DBSCAN]) -> list:
        if(type(model[0]) == DBSCAN):
            return [
                silhouette_score(X=x[i], labels=model[i].fit_predict(x[i])) 
                for i in range(len(x))
            ]
        elif(type(model[0]) == KMeans):
            return [
                silhouette_score(X=x[i], labels=model[i].predict(x[i])) 
                for i in range(len(x))
            ]

    @staticmethod
    def scoring(x=DataFrame([]),model=KMeans|DBSCAN) -> float:
        return silhouette_score(X=x, labels=model.predict(x))

class DaviesBouldinScore:
    def __init__(self, x=DataFrame([]), model=KMeans|DBSCAN):
        self.x = x
        self.model = model

    @staticmethod 
    def batch_scoring(x=[DataFrame([])], model=[KMeans|DBSCAN]) -> list:
        if(type(model[0]) == DBSCAN):
            return [
                davies_bouldin_score(X=x[i], labels=model[i].fit_predict(x[i])) 
                for i in range(len(x))
            ]
        elif(type(model[0]) == KMeans):
            return [
                davies_bouldin_score(X=x[i], labels=model[i].predict(x[i])) 
                for i in range(len(x))
            ]

    @staticmethod
    def scoring(x=DataFrame([]), model=KMeans | DBSCAN) -> float:
        return davies_bouldin_score(X=x, labels=model.predict(x))
    
