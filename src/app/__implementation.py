from pandas import DataFrame
from .__data_retrieve import RetrieveDataset
from .__model import KmeansModel, DbScanModel, ModelGeneration
from .__data_prepocessing import ProcessData
from .__crawler import Crawlers
from numpy import ndarray

class JobRecomendation:
    def __init__(
        self, dataset="", retrieval_types="file"):
        """
        this model would generate properties that gonna used on the 
        job recomendation model

        <h2>Parameters</h2>

        dataset = name of the dataset file, or the dataframe

        retrieval_types = how the data would get retrieved, neither by 
        take the file at root/src/data folder, or by directly put dataframe
        on it

        filter_on_each_column = list name of the column that gonna be used 
        for label encoding, list filled with a string

        filters_dict_in_list = list dict of the filter that used for label
        encoding

        dropped_columns = list of column that would get dropped

        one_hot_column = list of column that would get one hot encoding

        mean_numeric = list of column that would get the mean of a numeric 
        column (see the function at ProcessData class)

        seperators = list of seperators for mean_numeric

        model_type = model type that would get used (kmeans, dbscan)

        dimensional_reduction_type = dimensional reduction method that used
        (autoencoder, pca)

        <h2>optonal parameter</h2> 

        n_cluster = only used when the model use Kmeans model 

        min_sample = only used when the model use DBSCAN model

        hidden_layer = the list of hidden layer in the autoencoder
        
        loss = loss that used for evaluate the autoencoder 
        (see the model in transformer class) 
        
        epoch = epoch for autoencoder training
        
        to_shape = reduction of the autoencoder
        
        n_components = reduction of the pca method 
        """
        self.dataset = dataset

        self.retrieval_types = retrieval_types

        self.df = None

        self.retrieve_dataset = RetrieveDataset

        self.process_data = ProcessData

        self.models = ModelGeneration

        self.result_on_each_filter = []

        self.filter_column_name = [
            "experience_level","company_size","employment_type"]
        
        self.data_size = 0
        
        self.filters_dict_in_list = [
                {"Entry":1, "Mid":2, "Senior":3}, 
                {'Startup':1, 'Mid':2, 'Large':3},
                {'Internship':1,'Contract':2, 'Full-time':3, 'Remote':4}
            ]
        
        self.dropped_columns = [
        "skills_required", "tools_preferred", "industry", 
        "job_title", "salary_range_usd", 
        "company_name", "location", "posted_date", "job_id"]
        
        self.one_hot_column = ["skills_required", "tools_preferred", 
                               "industry", "job_title"]
        
        self.mean_numeric = ["salary_range_usd"]

        self.separators = ["-"]

        self.datetime_convertion = []

        self.model_type = "kmeans"

        self.dimensional_reduction_type = "pca"

        self.n_cluster = 3

        self.min_sample = 5

        self.hidden_layer = [32 ,16, 8]

        self.loss = ""

        self.epoch = 5

        self.to_shape = 4

        self.n_components = 4

    def initiate(self):
        """
        seperate function to initiate job recomendation model
        """
        match self.retrieval_types:
            case "file":
                try:
                    self.df = self.retrieve_dataset.get_DataFrame(
                        "src/data/" + self.dataset)
                except:
                    print("File Not Founded")
            case "dataframe":
                self.df = self.dataset

        self.result = self.df.copy()

        self.df = self.__first_step(self.df)

        self.__last_step()
    
    def __first_step(self,df=DataFrame([])):
        """
        first step of the algorithm (preprocessing)

        <h2>Parameters</h2>

        df = dataframe that gonna be processed

        <h2>Return</h2>

        dataframe that already get processed
        """
        list_one_hot_data = self.process_data.one_hot_encoder(
            df=df, feature_list=self.one_hot_column, 
            dataframe_result=self.one_hot_column
        )

        list_of_datetime_convertion = self.process_data.batch_datetime_convert(
            df=df, columns=self.datetime_convertion)

        mean_columns = self.process_data.batch_mean_processor(
            df=df, columns=self.mean_numeric, sep=self.separators)

        used_df = self.df.copy()
        
        used_df = self.process_data.batch_label_encoding_process(
            df=used_df, columns=self.filter_column_name, filters=self.filters_dict_in_list
        )

        used_df = self.process_data.dropColumns(df=used_df, columns=self.dropped_columns)

        combined_data = DataFrame([])

        for i in list_one_hot_data:
            combined_data = self.process_data.combineDataFrame(
                [combined_data, i], axis=1)
            
        processed_dataframe = [
            used_df, combined_data, list_of_datetime_convertion, mean_columns
        ]

        flatters = Crawlers()

        flatters.flatten_list(processed_dataframe)

        processed_dataframe = flatters.get_result()

        used_df = self.process_data.combineDataFrame(
            processed_dataframe, axis=1)
        
        self.employment_datas = JobRecomendation.applied_filters_to_dataframe(
            df=used_df,
            func=self.generate_list_from_dict,
            columns = self.filter_column_name, 
            filters = self.filters_dict_in_list
        )

        self.data_size = len(self.employment_datas)

        return used_df
    
    @staticmethod
    def applied_filters_to_dataframe(
        df=DataFrame([]),func=lambda x: x, 
        columns=[str], filters=[dict]) -> list:
        """
        function to applied filters to dataframe

        <h2>Parameters</h2>

        df = dataframe that want to get processed

        func = function that applied the filters 
        (generate_list_from_dict from JobRecomendation class)

        columns = list of string that filled with columns name

        filters = list of dict that act as filter for columns

        <h2>Return</h2>

        flattened result of the filtered df
        """
        result = JobRecomendation.combined_filters(
            df=df, func=func, columns=columns, filters=filters
        )

        crawler_obj = Crawlers()

        crawler_obj.flatten_list(result)

        return crawler_obj.get_result()

    @staticmethod
    def generate_list_from_dict(
            df=DataFrame([]), column="", filters=dict):
        """
        function to generate a list from the dictionary

        <h2>Parameters</h2>

        df = dataframe that want to be processed

        columns = column that will get processed

        filters = dict that act as label

        <h2>Return</h2>

        list of the filtered label, seperate by index
        """
        return [df[df[column] == i].copy().drop(
                [column], axis=1) for i in filters.values()]
    
    @staticmethod
    def combined_filters(
            df=DataFrame([]),func=generate_list_from_dict, columns=[str], 
            filters=[dict]):
        """
        function to combine all the processed list into one big flattened list

        <h2>Parameters</h2>
        
        df = dataframe that want to be processed

        func = function that will used as executor 
        (generate_list_from_dict from JobRecomendation)

        columns = list of column name that will be processed

        filters = list of dict that act as filter for label encoding

        <h2>Return</h2>

        flattened list of processed df
        """
        result = None
        if(len(columns) == len(filters)):
            if(len(columns) == 1):
                result = func(df, columns[0], filters[0])
            else:
                result = func(df, columns[0], filters[0])
                obj = Crawlers()
                columns = columns[1:]
                filters = filters[1:]
                for i in range(len(columns)):
                    obj.flatten_list()
                    obj.nested_list_filters_applicator(
                        x=result, func=func, column=columns[i], filters=filters[i]
                    )
                    result = obj.get_result()
                    obj.flush()
        return result

    def __last_step(self):
        """
        last step of the algorithm (model implementation)
        """
        index_data_list = [i.index for i in self.employment_datas]

        self.scaler = self.process_data.scalling(
            self.df.copy().drop(self.filter_column_name, 
            axis=1), "Min Max Scaler")

        self.autoencoder = self.process_data.dimensional_reduction_generation(
            x=self.df.copy().drop(self.filter_column_name, 
            axis=1), scaler=self.scaler, types=self.dimensional_reduction_type,
            hidden_layer=self.hidden_layer, loss=self.loss, epoch=self.epoch,
            to_shape=self.to_shape, n_components=self.n_components)

        match self.model_type:
            case "kmeans":
                self.model_list = self.models.model_generation(
                    x=self.employment_datas, 
                    preprocessing_implementation=self.preprocessing_implementation,
                    model_type=self.model_type, 
                    n_cluster=self.n_cluster)
            case "dbscan":
                self.model_list = self.models.model_generation(
                    x=self.employment_datas,
                    preprocessing_implementation=self.preprocessing_implementation,
                    model_type=self.model_type,
                    min_sample=self.min_sample
                )

        result_respect_to_filter = [self.__model_applicator(
                self.employment_datas[i], self.model_list[i]
            ) for i in range(self.data_size)]

        result_on_each_filter = [
            self.process_data.combineDataFrame(
                [self.result.iloc[index_data_list[i]], 
                    DataFrame(data=result_respect_to_filter[i], 
                        columns=["label"],index=index_data_list[i])],
                axis=1) for i in range(self.data_size)]

        self.result = self.process_data.combineDataFrame(
            result_on_each_filter, axis=0).sort_index()

    def __model_applicator(self,x=DataFrame([]),
            model=KmeansModel|DbScanModel) -> ndarray:
        """
        model prediction result 

        <h2>Parameters</h2>

        x = dataframe that want to get predicted

        model = model used for the algorithm (Kmeans, DBSCAN)

        <h2>Return</h2>

        a numpy result of the prediction
        """
        return model.predict(self.preprocessing_implementation(x))
    
    def get_filters(self, filters_index=0) -> dict:
        """
        get the filters base on index 
        (see self.filters_dict_in_list at JobRecomendation)

        <h2>Parameters</h2>

        filters_index = filter that want to retrieved

        <h2>Return</h2>
        
        filter in the dict form
        """
        return self.filters_dict_in_list[filters_index].keys()
    
    def preprocessing_implementation(
            self, x=DataFrame([])) -> ndarray:
        """
        function to implement preprocessing to the dataframe
        (scaling -> autoencoder -> numpy array)

        <h2>Parameters</h2>

        x = dataframe that want to get processed

        <h2>Return</h2>

        numpy array filled with result of preprocessing
        """
        return self.autoencoder.get_result(
            self.scaler.transform(x))
    
    def getData(self, filters=[str], hardness_level=None):
        """
        function to get result of model classifiation

        <h2>Parameters</h2>
        
        filters = list of string that will act as filter

        hardness_level = int value of the label from model prediction,
        none if want to get all the prediction in that specific filters

        <h2>Return</h2>

        Dataframe result that filtered with hardness level selection
        """
        result = self.result
        if(len(filters) == len(self.filter_column_name)):
            for i in range(len(filters)):
                print(result)
                result = result[
                    result[self.filter_column_name[i]] == filters[i]]
        if(hardness_level != None):
            try:
                result = result[result["label"] == hardness_level]
            except:
                result = "that employment type didn't exist"
        return result