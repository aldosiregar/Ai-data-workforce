from .__implementation import JobRecomendation

class Crawlers:
    def __init__(self):
        self.result = []
        self.index = 0

    def flush(self):
        """
        reset the 
        """
        self.result = []
        self.index = 0

    def get_result(self):
        """
        a function to get the result of crawlers

        return :

        result = flattened list of result
        """
        return self.result
    
    def flatten_list(self, x=[]):
        """
        a function to flatten the nested list

        parameter :
        x = a nested list
        """
        for i in x:
            if(type(i) == list):
                self.flatten_list(i)
            else:
                self.result.append(i)
