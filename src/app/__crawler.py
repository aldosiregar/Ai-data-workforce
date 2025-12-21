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

    def nested_list_filters_applicator(
            self, x=[], func=any, column="", filters=dict):
        """
        a function to flatten the nested list

        parameter :
        x = a nested list
        """
        for i in x:
            if(type(i) == list):
                self.nested_list_filters_applicator(
                    x=i, func=func, column=column, filters=filters)
            else:
                self.result.append(
                    func(i, column, filters)
                )