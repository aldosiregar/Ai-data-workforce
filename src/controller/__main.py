from ..ui import MainPage
from streamlit import navigation, cache_resource
from ..app import JobRecomendation

class PageNavigation:
    def __init__(self, dataset="", retrieval_types=""):
        """
        initiate navigation section
        """
        #pass the prediction model to main page
        self.Job_recomendation = InitiatedModel.initiated_model(
            dataset=dataset,retrieval_types=retrieval_types)

        all_pg = navigation([self.Job_recomendation_page])

        all_pg.run()

    def Job_recomendation_page(self):
        #the function that will generate the main page
        MainPage(self.Job_recomendation)

class InitiatedModel:
    """
    initiated model and put it in cache

    parameter : 

    filename : str

    return :

    obj : JobRecomendation class
    """
    @cache_resource
    def initiated_model(dataset="", retrieval_types="file"):
        obj = JobRecomendation(
            dataset=dataset, retrieval_types=retrieval_types)
        obj.initiate()
        return obj