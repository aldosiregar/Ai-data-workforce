from ..ui import MainPage
from streamlit import navigation, markdown, cache_resource
from ..app import JobRecomendation

class PageNavigation:
    def __init__(self, filename=""):
        self.Job_recomendation = InitiatedModel.initiated_model(filename)

        all_pg = navigation([self.Job_recomendation_page, self.pg2])

        all_pg.run()

    def Job_recomendation_page(self):
        MainPage(self.Job_recomendation)

    def pg2(self):
        markdown(f"""<h1>test 2</h1>""", unsafe_allow_html=True)

class InitiatedModel:
    @cache_resource
    def initiated_model(filename=""):
        obj = JobRecomendation(filename=filename)
        obj.initiate()
        return obj