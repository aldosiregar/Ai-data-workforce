import streamlit as st
from src.app import JobRecomendation

class MainPage:
    def __init__(self, job_recomendation=JobRecomendation):
        self.job_recomendation = job_recomendation

        employment_level, hardness_level = self.filter_generation()

        self.render(self.Main_page())

        if(st.button("search")):
            self.show_result(employment_level, hardness_level)

    def render(self,fstream=f""""""):
        st.markdown(fstream, unsafe_allow_html=True)

    def Main_page(self):
        return (f"""<div><h1>test aja</h1></div>""")
    
    def show_result(self,employment_level="",hardness_level=""):
        result = None
        match hardness_level:
            case "All":
                result = self.job_recomendation.getData(
                    employment_type=employment_level, hardness_level=None)
            case "Lowest":
                result = self.job_recomendation.getData(
                    employment_type=employment_level, hardness_level=0)
            case "Middle":
                result = self.job_recomendation.getData(
                    employment_type=employment_level, hardness_level=1)
            case "Highest":
                result = self.job_recomendation.getData(
                    employment_type=employment_level, hardness_level=2)

        st.write(result)

    def filter_generation(self):
        employment_level = st.selectbox(
            label="Employment Level",
            options=[
                "Internship",
                "Contract",
                "Full-time",
                "Remote"
            ])
        
        hardness_level = st.selectbox(
            label="Hardness Level",
            options=[
                "All",
                "Lowest",
                "Middle",
                "Highest"
            ])
        
        return employment_level, hardness_level