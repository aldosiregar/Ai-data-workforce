from pandas import DataFrame
from .__data_retrieve import RetrieveDataset
from .__model import KmeansModel, DbScanModel
from .__data_prepocessing import ProcessData
from .__autoencoder import Transformer
from .__crawler import Crawlers
from numpy import ndarray

class JobRecomendation:
    def __init__(self, filename=""):
        self.filename = filename

        self.df = None

        self.preprocessing = Preprocessing

        self.retrieve_dataset = RetrieveDataset

        self.process_data = ProcessData

        self.result_on_each_filter = []

        self.filter_column_name =[
            "experience_level", "company_size","employment_type"]
        
        self.data_size = 0

        self.experience_level_hirarchy = {
            "Entry":1, "Mid":2, "Senior":3}

        self.company_size_hirarchy = {
            'Startup':1, 'Mid':2, 'Large':3}

        self.employment_type_hirarchy = {
            'Internship':1,'Contract':2, 'Full-time':3, 'Remote':4}
        
        self.filters_dict_in_list = [
                self.experience_level_hirarchy, self.company_size_hirarchy,
                self.employment_type_hirarchy
            ]
        
        self.dropped_columns = [
        "skills_required", "tools_preferred", "industry", 
        "job_title", "salary_range_usd", 
        "company_name", "location", "posted_date", "job_id"]
        
        self.one_hot_column = ["skills_required", "tools_preferred", 
                               "industry", "job_title"]
        
        self.mean_numeric = ["salary_range_usd"]

        self.model_type = "kmeans"

        self.n_cluster = 3

        self.min_sample = 5

    def initiate(self):
        try:
            self.df = self.retrieve_dataset.get_DataFrame(
                "src/data/" + self.filename)
        except:
            print("File Not Founded")

        self.result = self.df.copy()

        self.df = self.__first_step(self.df)

        self.__last_step()
    
    def __first_step(self,df=DataFrame([])):
        list_one_hot_data = self.preprocessing.one_hot_encoder(
            df=df, feature_list=self.one_hot_column, 
            dataframe_result=self.one_hot_column
        )

        salary_range_usd = self.preprocessing.mean_of_columns(
            df=df, used_columns=self.mean_numeric)

        used_df = self.df.copy()
        
        used_df = self.preprocessing.batch_label_encoding_process(
            df=used_df, columns=self.filter_column_name, filters=self.filters_dict_in_list
        )

        used_df = self.process_data.dropColumns(df=used_df, columns=self.dropped_columns)

        combined_data = DataFrame([])

        for i in list_one_hot_data:
            combined_data = self.process_data.combineDataFrame(
                [combined_data, i], axis=1)

        used_df = self.process_data.combineDataFrame(
            [used_df, combined_data], axis=1)

        used_df = self.process_data.combineDataFrame(
            [used_df, salary_range_usd], axis=1)

        self.employment_datas = self.combined_filters(
            df=used_df,
            func=self.generate_list_from_dict,
            columns = self.filter_column_name, 
            filters = self.filters_dict_in_list
        )

        crawler_obj = Crawlers()

        crawler_obj.flatten_list(self.employment_datas)

        self.employment_datas = crawler_obj.get_result()

        self.data_size = len(self.employment_datas)

        return used_df
    
    def generate_list_from_dict(
            self,df=DataFrame([]), column="", filters=dict):
        return [df[df[column] == i].copy().drop(
                [column], axis=1) for i in filters.values()]
    
    def combined_filters(
            self,df=DataFrame([]),func=generate_list_from_dict, columns=[str], 
            filters=[dict]):
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
        index_data_list = [i.index for i in self.employment_datas]

        self.scaler = self.process_data.scalling(
            self.df.copy().drop(self.filter_column_name, 
            axis=1), "Min Max Scaler")

        self.autoencoder = self.preprocessing.dimensional_reduction_generation(
            x=self.df.copy().drop(self.filter_column_name, 
            axis=1), scaler=self.scaler, types="autoencoder")

        match self.model_type:
            case "kmeans":
                self.model_list = self.preprocessing.model_generation(
                    x=self.employment_datas, 
                    preprocessing_implementation=self.preprocessing_implementation,
                    model_type=self.model_type, 
                    n_cluster=self.n_cluster)
            case "dbscan":
                self.model_list = self.preprocessing.model_generation(
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
            model=KmeansModel):
        return model.predict(self.preprocessing_implementation(x))
    
    def get_employment_type_keys(self) -> dict:
        return self.employment_type_hirarchy.keys()
    
    def get_company_size_keys(self) -> dict:
        return self.company_size_hirarchy.keys()
    
    def get_experience_level_keys(self) -> dict:
        return self.experience_level_hirarchy.keys()
    
    def preprocessing_implementation(
            self, x=DataFrame([])) -> DataFrame | ndarray:
        return self.autoencoder.get_result(
            self.scaler.transform(x))
    
    def getData(self, filters=[str], hardness_level=None):
        result = self.result
        if(len(filters) == len(self.filter_column_name)):
            for i in range(len(filters)):
                result = result[
                    result[self.filter_column_name[i]] == filters[i]]
        if(hardness_level != None):
            try:
                result = result[result["label"] == hardness_level]
            except:
                result = "that employment type didn't exist"
        return result
    
class Preprocessing:
    @staticmethod
    def one_hot_encoder(
        df=DataFrame([]), feature_list=[str], dataframe_result=[str]):
        return [ProcessData.oneHotEncoding(df[feature_list[i]],
                dataframe_name=dataframe_result[i]) for i in range(len(
                    feature_list))]
    
    @staticmethod
    def mean_of_columns(df=DataFrame([]), used_columns="") -> DataFrame:
        """
            split one columns that still seperated by - \n
            example : 1000-2999

            return : DataFrame, else void
        """
        if(type(used_columns) == str):
            #get the data that want to be splitted
            salary_range_usd = df[used_columns].copy()
            return DataFrame(DataFrame(salary_range_usd.map(
                arg=lambda x: [int(i) for i in str.split(x, sep="-")]
            ).to_list(), columns=[
                "min_range_" + used_columns, 
                "max_range_slary" + used_columns]).mean(
                axis=1), columns=["mean_" + used_columns])
        else:
            print("not string, cant be used")

    @staticmethod
    def label_encoding(
        df=DataFrame([]), filter_dict=dict) -> DataFrame:
        return ProcessData.labelEncoding(
            df=df, format=filter_dict)
    
    @staticmethod
    def batch_label_encoding_process(
        df=DataFrame([]), columns=[str], 
        filters=[dict]) -> DataFrame:
        for i in range(len(columns)):
            df[columns[i]] = Preprocessing.label_encoding(
                df[columns[i]],filter_dict=filters[i])
        return df
    
    @staticmethod
    def dimensional_reduction_generation(
        x=DataFrame([]),
        scaler=ProcessData.scalling, types="pca"):
        result = x
        match types:
            case "autoencoder":
                hidden_layer =  [32 ,16, 8]
                loss = ""
                epoch = 5
                to_shape = 4

                result = Transformer.transform(
                scaler.transform(result), input_shape=x.shape[1], 
                to_shape=to_shape, hidden_layer=hidden_layer, 
                loss=loss, epoch=epoch)
            case "pca":
                n_components = 4

                result = ProcessData.pcaDimentionalityReduction(
                    scaler.transform(result), n_components=n_components
                )

        return result
    
    @staticmethod
    def model_generation(
        x=[], 
        preprocessing_implementation=JobRecomendation.preprocessing_implementation,
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
                        n_cluster=n_cluster) for i in x]
                
        return result