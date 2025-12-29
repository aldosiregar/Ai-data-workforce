import streamlit as st
from src.app import JobRecomendation

class MainPage:
    def __init__(self, job_recomendation=JobRecomendation):
        self.job_recomendation = job_recomendation

        self.render(self.start_page())

        #get the dict from the model, for generate filter
        self.experience_level_dict = job_recomendation.get_filters(0)
        self.company_size_dict = job_recomendation.get_filters(1)
        self.employment_type_dict = job_recomendation.get_filters(2)

        (employment_type, company_size, 
        experience_level,hardness_level) = self.filter_generation()

        self.render(self.results_section())

        #if button clicked
        if(st.button("search")):
            self.show_result(
                employment_type, company_size, experience_level,
                hardness_level)

    def start_page(self):
        return (
            f"""
            <div>
                <h1>Job Recomendation Model</h1>
                <p>
                    this model would filter the recomendation of job aplication
                    based on the type of filter below, the result may represent in
                    number, from 0 for the easiest, and 2 for the hardest.
                    <br><br>
                    you also can use all if you want to see all the result based
                    on the filter selected.
                </p>
                <h2>Filters : </h2>
            </div>
            """)

    def render(self,fstream=f""""""):
        """
        used to render any user interaction from the fstring
        """
        st.markdown(fstream, unsafe_allow_html=True)

    def results_section(self):
        return (f"""<div><h2>results : </h2></div>""")
    
    def show_result(
            self,employment_type="", company_size="", experience_level="",
            hardness_level=""):
        """
        function for show the results of model prediction

        parameter :

        employment_type = type of employment filter

        company_size = size of company filter

        experience_level = level of experience filter

        hardness_level = level of job difficulty 
        """
        result = None
        match hardness_level:
            case "All":
                result = self.job_recomendation.getData(
                    filters=[experience_level,company_size,employment_type],
                    hardness_level=None)
            case "Lowest":
                result = self.job_recomendation.getData(
                    filters=[experience_level,company_size,employment_type],
                    hardness_level=0)
            case "Middle":
                result = self.job_recomendation.getData(
                    filters=[experience_level,company_size,employment_type],
                    hardness_level=1)
            case "Highest":
                result = self.job_recomendation.getData(
                    filters=[experience_level,company_size,employment_type],
                    hardness_level=2)

        st.write(result)

    def filter_generation(self):
        """
        function to generate filter ui

        return :

        employment_type = str

        experience_level = str

        company_level = str
        """
        employment_type = st.selectbox(
            label="Employment Level",
            options=[
                i for i in self.employment_type_dict
            ])

        company_level = st.selectbox(
            label="Company Level",
            options=[
                i for i in self.company_size_dict
            ]
        )

        experience_level = st.selectbox(
            label="Experience Level",
            options=[
                i for i in self.experience_level_dict
            ]
        )
        
        hardness_level = st.selectbox(
            label="Hardness Level",
            options=[
                "All",
                "Lowest",
                "Middle",
                "Highest"
            ])
        
        return employment_type, company_level, experience_level, hardness_level