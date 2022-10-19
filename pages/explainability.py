# Import necessary libraries



from pathlib import Path
import streamlit as st
from PIL import Image

import sys
sys.path.append('..')

from lib.multiapp import MultiPage
from lib.page_functions import load_image



DataFolder = Path("./assets/")

# Show the header image

def app():

    """
        This function in this page is run when the page is invoked in the sidebar
    """    
    project_image = load_image("page_header.jpg")
    summary_plot_image  = load_image("rf_summary_plot.png")
    dependency_plot_image = load_image("rf_dependency_plot.png")
    force_plot_image = load_image("rf_force_plot.png")
    
    st.image(project_image, use_column_width=True)

    st.subheader(
        "Explainability through SHAP"
    )
    col1, col2 = st.columns(2) 
    with col1:
        st.image(summary_plot_image, use_column_width=True)
        st.caption("Summary Plot")
    
    with col2:
        st.image(dependency_plot_image, use_column_width=True)
        st.caption("Dependency Plot")

    st.markdown("""We used **SH**apley **A**dditive Ex**P**lanations to explain the
predictions of the RandomForestRegressor by generating summary plot,
dependency plot and force plots. The SHAP model is trained on the
transformed imputed train set which is the background distribution. The
tests are conducted on the test set on which we predict values. We have
the choice to predict on singular instances or multiple instances. The
summary plot (Fig. 6) shows current depth as the most dominant predictor
followed by arid soils. The dependency plot (Fig. 7) further emphasizes
the relation between increasing current depth and arid soils.""")
    
    
    st.image(force_plot_image, use_column_width=True)
    st.caption("Force Plot")

    st.markdown("""For the force plot (Fig. 8) we pick an instance of over and under
prediction by sorting the prediction by the difference of prediction and
observed value in the test set. The third lowest under-prediction and
the fifth highest over-prediction instance are shown in figure. The base
value of 11.9 is the prediction when no feature is taken into account.
The plot will display features upon hover. Here we observe that lower
values of Groundsurface elevation, followed by arid soil, reduce the
value of the prediction in the under-prediction. Whereas the
over-prediction in this instance is fueled by Current Depth and arid
soils and reservoir capacity.

""", unsafe_allow_html=True)
    
    st.markdown("""---""")

