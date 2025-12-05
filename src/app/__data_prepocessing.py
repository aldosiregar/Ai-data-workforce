from pandas import DataFrame, to_datetime, concat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class ProcessData:
    @staticmethod
    def oneHotEncoding(df=DataFrame([]), dataframe_name="") -> DataFrame:
        """
        parameter : \n 
        df = DataFrame \n dataframe_name = str \n

        return : 
        DataFrame \n

        data format : ("A", "B", "C", "D")

        result : 
        """
        set_container = set()

        #this is for adding a 
        def adder(x=[]):
            for i in x:
                #too much hassle, so i just make a condition
                if(i[0] == ' '):
                    i = i[1:]
                set_container.add(i)
        

        seperator = lambda x: str.split(x, sep=",")
        
        df = df.map(arg=seperator)
        
        df.map(adder)
        
        template = dict(zip(set_container,[0 for _ in range(len(set_container))]))
        
        result_container = []
        
        def initiated_counter(x):
            initiated = template.copy()
            for i in x:
                #too much hassle, so i just make a condition
                if(i[0] == ' '):
                    i = i[1:]
                if(i in initiated):
                    initiated[i] += 1
            return list(initiated.values())
        
        df.map(arg=(
            lambda x: result_container.append(
                initiated_counter(x)
            )
        ))

        result = DataFrame(result_container, columns=list(set_container))

        result.columns.name = dataframe_name
        
        return result
    
    @staticmethod
    def datetimeConvert(df=DataFrame([]), column="", format="%Y-%M-%D"):
        return to_datetime(df[column], format="%Y-%m-%d").copy()
    
    @staticmethod
    def dropColumns(df=DataFrame([]), columns=[]):
        return df.drop(columns=columns).copy()
    
    @staticmethod
    def pcaDimentionalityReduction(
        df=DataFrame([]),n_components=3) -> tuple:
        """
        return : tuple (numpy, PCA)
        """
        decomposer = PCA(n_components=n_components)
        decomposer.fit(df)
        return (
            DataFrame(
                decomposer.fit_transform(df), columns=["pca_" + str(df.columns.name)]), 
            decomposer)
    
    @staticmethod
    def combineDataFrame(data=DataFrame([]), axis=1) -> DataFrame:
        return concat([data],axis=axis)

    @staticmethod
    def labelEncoding(df=DataFrame([]),format=dict) -> DataFrame:
        return df.map(format)
    
    @staticmethod
    def scalling(df=DataFrame([]), type="") -> tuple:
        scaler = None
        match type:
            case "Standard Scaler":
                scaler = StandardScaler()
            case "Min Max Scaler":
                scaler = MinMaxScaler()
            case "Robust Scaler":
                scaler = RobustScaler()
            case _:
                scaler = StandardScaler()
                
        scaler.fit(df)
        return (scaler.transform(df), scaler)