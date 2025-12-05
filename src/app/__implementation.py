from pandas import DataFrame, errors
from .__data_retrieve import RetrieveDataset
from .__model import KmeansModel
from .__data_prepocessing import ProcessData
from .__autoencoder import Transformer

class JobRecomendation:
    def __init__(self, filename=""):
        self.df = None
        try:
            self.df = RetrieveDataset.get_DataFrame("../data/" + filename)
        except errors as e:
            print(e)

        list_one_hot_data = [
            ProcessData.oneHotEncoding(
                df=self.df["skills_required"],dataframe_name="skill_required"),
            ProcessData.oneHotEncoding(
                self.df["tools_preferred"], dataframe_name="tools_preferred"),
            ProcessData.oneHotEncoding(
                self.df["industry"], dataframe_name="industry"),
            ProcessData.oneHotEncoding(
                self.df["job_title"], dataframe_name="job_title")]

        salary_range_usd = self.df["salary_range_usd"].copy()
        salary_range_usd = DataFrame(salary_range_usd.map(
            arg=lambda x: [int(i) for i in str.split(x, sep="-")]
        ).to_list(), columns=["min_range_salary", "max_range_slary"]).mean(
            axis=1, columns=["mean_salary_range"])

        used_df = self.df.copy()

        experience_level_hirarchy = {
            "Entry":0, "Mid":1, "Senior":2}

        company_size_hirarchy = {
            'Startup':0, 'Mid':1, 'Large':2}

        self.employment_type_hirarchy = {
            'Internship':0,'Contract':1, 'Full-time':2, 'Remote':3}

        used_df["experience_level"] = ProcessData.labelEncoding(
            used_df["experience_level"], experience_level_hirarchy)

        used_df["company_size"] = ProcessData.labelEncoding(
            used_df["company_size"], company_size_hirarchy)

        used_df["employment_type"] = ProcessData.labelEncoding(
            used_df["employment_type"], self.employment_type_hirarchy)
        
        dropped_columns = [
        "skills_required", "tools_preferred", "industry", 
        "job_title", "salary_range_usd", 
        "company_name", "location", "posted_date", "job_id"]

        used_df = ProcessData.dropColumns(df=used_df, columns=dropped_columns)

        list_one_hot_data = ProcessData.combineDataFrame(
            list_one_hot_data, axis=1)

        used_df = ProcessData.combineDataFrame(
            [used_df, list_one_hot_data], axis=1)

        used_df = ProcessData.combineDataFrame(
            [used_df, salary_range_usd], axis=1)

        employment_datas = [
            used_df[used_df["employment_type"] == 0].copy().drop(
                "employment_type", axis=1), 
            used_df[used_df["employment_type"] == 1].copy().drop(
                "employment_type", axis=1), 
            used_df[used_df["employment_type"] == 2].copy().drop(
                "employment_type", axis=1), 
            used_df[used_df["employment_type"] == 3].copy().drop(
                "employment_type", axis=1)]
        
        index_data_list = [i.index for i in employment_datas]

        self.scaler_list = []

        index = 0

        for i in index_data_list:
            scalled_data, scaler = ProcessData.scalling(i)
            employment_datas[index] = scalled_data
            self.scaler_list.append(scaler)
            index += 1

        self.autoencoder_list = self.autoencoder_generation(
            employment_datas)

        self.model_list = self.kmeans_generation(employment_datas)

        result_respect_to_employment_level = []

        index = 0

        for i in employment_datas:
            result_respect_to_employment_level.append(
                self.model_list[index].predict(
                    self.preprocessing_implementation(i, index=index)))
            index += 1

        self.result_on_each_employment_level = []

        employment_type_list = [
            i for i in self.employment_type_hirarchy.keys()]

        index = 0

        result = self.df.copy()

        for i in result_respect_to_employment_level:
            self.result_on_each_employment_level.append(
                ProcessData.combineDataFrame(
                    [result[result[
                        "employment_type"] == employment_type_list[index]],
                    DataFrame(
                        self.preprocessing_implementation(i, index=index), 
                        columns=["label"], index=index_data_list[index])]
                , axis=1))
            index += 1

    def getData(self, employment_type="", hardness_level=None):
        result = None
        if(hardness_level):
            try:
                result = self.result_on_each_employment_level[
                    self.employment_type_hirarchy[
                        employment_type]][self.employment_type_hirarchy[
                            employment_type] == hardness_level]
            except:
                result = "that employment type didn't exist"
        else:
            try:
                result = self.result_on_each_employment_level[
                    self.employment_type_hirarchy[employment_type]]
            except:
                result = "that employment type didn't exist"
        return result
    
    def autoencoder_generation(self,x=[]):
        hidden_layer =  [32 ,16, 8]
        loss = ""
        epoch = 5
        to_shape = 4

        autoencoder_list = []

        for i in x:
            autoencoder = Transformer.transform(
            i, input_shape=i.shape[1], to_shape=to_shape, hidden_layer=hidden_layer, 
            loss=loss, epoch=epoch)
            autoencoder_list.append(autoencoder)

        return autoencoder_list
    
    def kmeans_generation(self,x=[]):
        index = 0

        kmeans_list = []
        
        for i in x:
            kmeans_list.append(KmeansModel(
                df=self.preprocessing_implementation(
                    i, index=index), n_cluster=3))
         
        return kmeans_list
    
    def preprocessing_implementation(self, x=DataFrame([]), index=0):
        return self.autoencoder_list[index].get_result(
            self.scaler_list[index].transform(x))