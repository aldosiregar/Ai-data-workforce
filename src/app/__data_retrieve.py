from pandas import read_csv, DataFrame

class RetrieveDataset:
    @staticmethod
    def get_DataFrame(filepath=str) -> DataFrame:
        return read_csv(filepath)