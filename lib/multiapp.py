import streamlit as st

# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        :params:title ([str]): The title of page which we are adding to the list of apps 
        :params:func: Python function to render this page in Streamlit
        """

        self.pages.append({"title": title, "function": func})

    def run(self):
        # Drodown to select the page to run
        page = st.sidebar.selectbox(
            "Groundwater Shortage Exploration", self.pages, format_func=lambda page: page["title"]
        )
               
        st.sidebar.markdown("")
        st.sidebar.markdown('##')
        st.sidebar.subheader("Contributors")
        st.sidebar.markdown("Simi Talkar and Matthieu Lienart")
        st.sidebar.markdown('##')
        st.sidebar.markdown('##')
        
        url_git = 'https://github.com/sjtalkar/milestone2_waterwells_deepnote'
        st.sidebar.markdown(f"Source Code on [Github]({url_git})")
        
        
        # run the app function
        page["function"]()
    


