from pandas import DataFrame
from .__data_retrieve import RetrieveDataset
from .__model import KmeansModel
from .__data_prepocessing import ProcessData
from .__autoencoder import Transformer

class JobRecomendation:
    def __init__(self, filename=""):
        self.df = None
        try:
            self.df = RetrieveDataset.get_DataFrame("src/data/" + filename)
        except:
            print("File Not Founded")

        self.result = self.df.copy()

        self.df = self.__first_step(self.df)

    def getData(self, employment_type="", hardness_level=None):
        result = None
        if(hardness_level != None):
            try:
                result = self.result_on_each_employment_level[
                    self.get_employment_hirarchy_label(
                        employment_type=employment_type)][
                            self.result_on_each_employment_level[
                                self.get_employment_hirarchy_label(
                                employment_type=employment_type)][
                                    "label"] == hardness_level]
            except:
                result = "that employment type didn't exist"
        else:
            try:
                result = self.result_on_each_employment_level[
                    self.employment_type_hirarchy[employment_type]]
            except:
                result = "that employment type didn't exist"
        return result
    
    def __first_step(self,df=DataFrame([])):
        list_one_hot_data = Preprocessing.one_hot_encoder(
            df=df, feature_list=[
                "skills_required", "tools_preferred", "industry", "job_title"
            ], dataframe_result=[
                "skills_required", "tools_preferred", "industry", "job_title" 
            ]
        )

        salary_range_usd = Preprocessing.mean_of_columns(
            df=df, used_columns="salary_range_usd")

        used_df = df.copy()

        self.experience_level_hirarchy = {
            "Entry":0, "Mid":1, "Senior":2}

        self.company_size_hirarchy = {
            'Startup':0, 'Mid':1, 'Large':2}

        self.employment_type_hirarchy = {
            'Internship':0,'Contract':1, 'Full-time':2, 'Remote':3}
        
        used_df = Preprocessing.batch_label_encoding_process(
            df=used_df, columns=[
                "experience_level", "company_size", "employment_type"
            ], filters=[
                self.experience_level_hirarchy, self.company_size_hirarchy,
                self.employment_type_hirarchy
            ]
        )
        
        dropped_columns = [
        "skills_required", "tools_preferred", "industry", 
        "job_title", "salary_range_usd", 
        "company_name", "location", "posted_date", "job_id"]

        used_df = ProcessData.dropColumns(df=used_df, columns=dropped_columns)

        combined_data = DataFrame([])

        for i in list_one_hot_data:
            combined_data = ProcessData.combineDataFrame(
                [combined_data, i], axis=1)

        used_df = ProcessData.combineDataFrame(
            [used_df, combined_data], axis=1)

        used_df = ProcessData.combineDataFrame(
            [used_df, salary_range_usd], axis=1)

        self.employment_datas = self.combined_filters(
            df=used_df,
            items=[
                self.generate_employment_type_list,
                self.generate_company_size_list,
                self.generate_experience_level_list
            ]
        )

        return used_df
    
    def generate_company_size_list(self,df=DataFrame([])):
        return [
            df[df["company_size"] == i].copy().drop(
                ["company_size"], axis=1
            ) for i in self.company_size_hirarchy.values
        ]
    
    def generate_experience_level_list(
        self,df=DataFrame([])) -> list:
        return [
            df[df["experience_level"] == i].copy().drop(
                ["experience_level"], axis=1
            ) for i in self.experience_level_hirarchy.values
        ]
    
    def generate_employment_type_list(
        self, df=DataFrame([])) -> list:
        return [
            df[df["employment_type"] == i].copy().drop(
                ["employment_type"], axis=1
            ) for i in self.employment_type_hirarchy.values
        ]
    
    def combined_filters(self,df=DataFrame([]),items=[]):
        if len(items) == 1 : return  items[0](df)
        result = items[0](df)
        for i in items[1:]:
            result = [
                i(j) for j in result
            ]
        return result


    def __last_step(self):
        obj = Crawlers()

        obj.normal_crawler(x=self.employment_datas)

        index_data_list = obj.get_result 

        obj.flush()

        self.scaler = ProcessData.scalling(self.df.copy().drop(
            "employment_type", axis=1), "Min Max Scaler")

        index = 0

        self.autoencoder = Preprocessing.autoencoder_generation(
            self.employment_datas, scaler=self.scaler)

        self.model_list = Preprocessing.kmeans_generation(
            self.employment_datas, self.preprocessing_implementation)

        result_respect_to_employment_level = []

        index = 0

        #next task : join crawler with this prediction

        for i in self.employment_datas:
            result_respect_to_employment_level.append(
                self.model_list[index].predict(
                    self.preprocessing_implementation(i)))
            index += 1

        self.result_on_each_employment_level = []

        employment_type_list = [
            i for i in self.employment_type_hirarchy.keys()]

        index = 0

        for i in result_respect_to_employment_level:
            self.result_on_each_employment_level.append(
                ProcessData.combineDataFrame(
                    [result[result[
                        "employment_type"] == employment_type_list[index]],
                    DataFrame(
                        i, 
                        columns=["label"], index=index_data_list[index])]
                , axis=1))
            index += 1
    
    def get_employment_hirarchy_label(self,employment_type=""):
        return self.employment_type_hirarchy[employment_type]
    
    def preprocessing_implementation(self, x=DataFrame([])):
        return self.autoencoder.get_result(
            self.scaler.transform(x))
    

class Crawlers:
    def __init__(self,n_cluster=3):
        self.n_cluster = n_cluster
        self.result = []

    def flush(self):
        self.temp = []

    def get_result(self):
        return self.result

    def crawler_kmeans(self,x=[], func=lambda x:x):
        for i in x:
            if(type(i) == list):
                self.crawler_kmeans(i, func=func)
            else:
                self.result.append(
                    KmeansModel(func(i), self.n_cluster))

    def normal_crawler(self,x=[], func=lambda x:x):
        for i in x:
            if(type(i) == list):
                self.normal_crawler(i, func=func)
            else:
                self.result.append(func(i))
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
        df=DataFrame([]), columns_name="", 
        filter_dict=dict) -> DataFrame:
        return ProcessData.labelEncoding(
            df[columns_name], filter_dict)
    
    @staticmethod
    def batch_label_encoding_process(
        df=DataFrame([]), columns=[str], 
        filters=[dict]) -> DataFrame:
        for i in range(len(columns)):
            df[columns[i]] = Preprocessing.label_encoding(
                df[columns[i]], columns_name=columns[i],
                filters=filters[i]
            )
        return df
    
    @staticmethod
    def autoencoder_generation(
        x=DataFrame([]),
        scaler=ProcessData.scalling):
        hidden_layer =  [32 ,16, 8]
        loss = ""
        epoch = 5
        to_shape = 4

        autoencoder = Transformer.transform(
        scaler.transform(x), input_shape=x.shape[1], 
        to_shape=to_shape, hidden_layer=hidden_layer, 
        loss=loss, epoch=epoch)

        return autoencoder
    
    @staticmethod
    def kmeans_generation(
        x=[], 
        preprocessing_implementation=lambda x:x,
        n_cluster=3):

        crawler = Crawlers(n_cluster=n_cluster)
        
        crawler.crawler_kmeans(x=x, func=preprocessing_implementation)
         
        return crawler.get_result()