from pandas import DataFrame, Series ,to_datetime, concat
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .__autoencoder import Transformer

class ProcessData:
    @staticmethod
    def oneHotEncoding(df=DataFrame([]), dataframe_name="") -> DataFrame:
        """
        <h2>Parameters</h2> 

        df = DataFrame of the data
        
        dataframe_name = the name that would be assign to the dataframe

        <h2>Return</h2> 
        
        DataFrame 

        data format = ("A", "B", "C", "D")
        """
        set_container = set()

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
    def datetimeConvert(
        df=Series([]), format="%Y-%M-%D") -> datetime:
        """
        convert DateTime from str to DateTime object

        example : 1991-11-20 (str) ->  1991-11-20 (DateTime)

        <h2>Parameters</h2>

        df = dataframe of the data that want to get changed

        column = the column that want to get changed

        format = the format of the DateTime

        <h2>Return</h2> 
        
        result of the data conversion
        """
        return to_datetime(df, format=format).copy()
    
    @staticmethod
    def batch_datetime_convert(df=DataFrame([]), columns=[]) -> list:
        """
        function for convert the list of columns into datetime

        <h2>Parameters</h2>

        df = dataframe that want to be altered

        columns = list of columns that want to get changed

        <h2>Return</h2>

        list of datetime that already got converted
        """
        return [
            ProcessData.datetimeConvert(
                df[columns[i]]) for i in range(len(columns))]
    
    @staticmethod
    def dropColumns(df=DataFrame([]), columns=[]) -> DataFrame:
        """
        drop a columns

        <h2>Parameters</h2>

        df = data that want to get altered

        columns = list of columns that want to get dropped

        <h2>Return</h2>

        result of dropped columns
        """
        return df.drop(columns=columns).copy()
    
    @staticmethod
    def pcaDimentionalityReduction(
        df=DataFrame([]),n_components=3) -> PCA:
        """
        function to reduce dimentionality (pca method)

        <h2>Parameters</h2>

        df = dataframe that want to get altered

        n_components = how small pca would decompose the features

        <h2>Return</h2> 
        
        PCA model that already fitted with the the df
        """
        decomposer = PCA(n_components=n_components)
        decomposer.fit(df)

        return decomposer
    
    @staticmethod
    def combineDataFrame(data=[], axis=1) -> DataFrame:
        """
        function for combined two or more dataframe

        <h2>Parameters</h2>

        data = column of the dataframe that want to get joined

        axis = the axis of the process (1 for column-wise, 0 for index-wise)

        <h2>Return</h2>

        the results of joined dataframe
        """
        return concat(data,axis=axis)

    @staticmethod
    def labelEncoding(df=([]),format=dict) -> DataFrame:
        """
        function for label encoding function

        <h2>Parameters</h2>

        df = dataframe that want to get altered

        <h2>Return</h2>

        label encoded DataFrame
        """
        return df.map(format)
    
    @staticmethod
    def scalling(df=DataFrame([]), type=""):
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
        return scaler
    
    @staticmethod
    def one_hot_encoder(
        df=DataFrame([]), feature_list=[str], dataframe_result=[str]):
        """
        
        """
        return [ProcessData.oneHotEncoding(df[feature_list[i]],
                dataframe_name=dataframe_result[i]) for i in range(len(
                    feature_list))]
    
    @staticmethod
    def mean_of_columns(
        df=Series([]), used_columns="", sep="-") -> DataFrame:
        """
        split one columns that still seperated by (-) 
        (,) or custom separator. for example : (1000-2999), (1000,2000)

        <h2>Parameters</h2>

        df = Series of the dataset that want to get converted 

        sep = seperator of the string (-), (,), or custom separator

        <h2>Return</h2> 
            
        series of the mean, else void
        """
        numeric_value = df
        return DataFrame(DataFrame(numeric_value.map(
            arg=lambda x: [int(i) for i in str.split(x, sep=sep)]
            ).to_list(), columns=[
                "min_range_" + used_columns, 
                "max_range_slary" + used_columns]).mean(
                    axis=1), columns=["mean_" + used_columns])

    @staticmethod
    def batch_mean_processor(
        df=DataFrame([]), columns=[], sep=["-"]) -> list:
        """
        function to do batch operation of searching mean of columns 
        (see mean_of_columns in ProcessData class)

        <h2>Parameters</h2>

        df = dataframe that want to be altered

        columns = list of columns that want to get processed

        sep = list of separator in the series (-), (,), or custom separator

        <h2>Return</h2>

        list of the processed columns
        """
        return [
            ProcessData.mean_of_columns(
                df=df[columns[i]],used_columns=columns[i], 
                sep=sep[i]) for i in range(len(columns))]

    @staticmethod
    def label_encoding(
        df=Series([]), filter_dict=dict) -> DataFrame:
        """
        function to do label encoding operation

        <h2>Parameters</h2>

        df = Series that want to get altered

        filter_dict = the filter of that label

        <h2>Return</h2>

        processed df that already get labeled based of filters
        """
        return ProcessData.labelEncoding(
            df=df, format=filter_dict)
    
    @staticmethod
    def batch_label_encoding_process(
        df=DataFrame([]), columns=[str], 
        filters=[dict]) -> DataFrame:
        """
        the batch operation of the label encoding

        <h2>Parameters</h2>

        df = the dataframe that want to get labeled

        columns = list of columns that want to get labeled

        filters = list of filters for the labeling purpose

        <h2>Return</h2>

        the dataframe that already get labeled
        """
        for i in range(len(columns)):
            df[columns[i]] = ProcessData.label_encoding(
                df[columns[i]],filter_dict=filters[i])
        return df
    
    @staticmethod
    def dimensional_reduction_generation(
        x=DataFrame([]),scaler=scalling, types="autoencoder",
        hidden_layer=[32 ,16, 8], loss="", epoch=5, to_shape=4,
        n_components=4):
        """
        function to generate model of dimensional reduction, the type that 
        provide is autoencoder and pca
        
        <h2>Parameters</h2>

        x = the dataframe that want to get altered
        
        scaler = scaler model that generated from scaling function in 
        ProcessData class
        
        types = the type of the dimensional reduction, autoencoder or pca

        <h2>Optional Parameter</h2>
        
        hidden_layer = the list of hidden layer in the autoencoder
        
        loss = loss that used for evaluate the autoencoder 
        (see the model in transformer class) 
        
        epoch = epoch for autoencoder training
        
        to_shape = reduction of the autoencoder
        
        n_components = reduction of the pca method 

        <h2>Return</h2>

        model of the dimensional reduction
        """
        result = x
        match types:
            case "autoencoder":
                result = Transformer.transform(
                scaler.transform(result), input_shape=x.shape[1], 
                to_shape=to_shape, hidden_layer=hidden_layer, 
                loss=loss, epoch=epoch)
            case "pca":
                result = ProcessData.pcaDimentionalityReduction(
                    scaler.transform(result), n_components=n_components
                )

        return result