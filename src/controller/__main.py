from ..ui import MainPage
from streamlit import navigation, markdown, cache_resource
from ..app import JobRecomendation

class PageNavigation:
    def __init__(self, filename=""):
        """
        initiate page ui
        """

        #pass the prediction model to main page
        self.Job_recomendation = InitiatedModel.initiated_model(filename)

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
    def initiated_model(filename=""):
        obj = JobRecomendation(filename=filename)
        obj.initiate()
        return obj