from pandas import read_csv, DataFrame

class RetrieveDataset:
    def __init__(self, filepath=str):
        self.df = read_csv(filepath)
    
    def get_DataFrame(self) -> DataFrame:
        return self.df
    
    def copy_DataFrame(self) -> DataFrame:
        return self.df
    
    @staticmethod
    def get_DataFrame(filepath=str) -> DataFrame:
        return read_csv(filepath)