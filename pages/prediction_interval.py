# Import necessary libraries

import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
from lib.township_range import TownshipRanges
from streamlit_folium import folium_static 

sys.path.append("..")

from lib.multiapp import MultiPage
from lib.page_functions import load_image, set_png_as_page_bg
from lib.uncertainty import create_prediction_uncertainty_qrf, create_prediction_uncertainty_gbr 


DataFolder = Path("./assets/")
# Load the model once created
# model = pickle.load(open("filename", "rb"))



def app():

    """
        This function in this page is run when the page is invoked in the sidebar
    """
    y_pred_df = None
    project_image = load_image("page_header.jpg")
    st.image(project_image, use_column_width=True)

    # Project Proposal
    ###########################################
    st.subheader("Prediction Intervals", anchor="prediction_interval")
           
    y_pred_df = pd.read_csv('./pages/predicting_uncertainty_gbr.csv', 
                                dtype={'lower':np.float64, 'mid': np.float64, 'upper': np.float64})
#     y_pred_qrf_df = pd.read_csv('./pages/predicting_uncertainty_qrf.csv',
#                                dtype={'lower':np.float64, 'mid': np.float64, 'upper': np.float64})
     
    
        
#     counties_list = list(y_pred_qrf_df.COUNTY.unique())
#     interval_list  = ['lower', 'mid', 'upper']
    township_range = TownshipRanges()
    county_tr_mapping = township_range.counties_and_trs_df
    y_pred_df = county_tr_mapping.merge(y_pred_df, left_on='TOWNSHIP_RANGE', right_on='TOWNSHIP_RANGE') 
      
    st.write(type(county_tr_mapping), type(y_pred_df))
    
    
    if st.button("Predict"):
            st.subheader(f"Groundwater Depth Predictions For Quantiles")
            st.caption(f"Township-Ranges in San Joaquin river basin")
            #st.caption(f"County: {county_selected}")
            st.caption(f"The prediction values for lower median and upper quantile")
            st.markdown("""---""")
            folium_static(y_pred_df.explore(column='lower', cmap='twilight_r'))
            st.markdown("""---""")
            #Streamlit cannot deal with geometry column
            new_df = y_pred_df.drop(columns=['geometry'])
            st.dataframe(new_df[['COUNTY', 'TOWNSHIP_RANGE', 'YEAR', model_selected]])
           
    st.markdown("""---""")

    