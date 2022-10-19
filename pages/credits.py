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
    st.image(project_image, use_column_width=True)
    st.subheader(
        "Credits",
        anchor="credits",
    )

    st.markdown("""---""")
    
    st.markdown("""
1.  Sustainable Groundwater Management Act (SGMA). *California Department of Water Resources*. [[https://water.ca.gov/programs/groundwater-management/sgma-groundwater-management]{.underline}](https://water.ca.gov/programs/groundwater-management/sgma-groundwater-management). Last Accessed 15th October 2022.
2.  M. L. La Ganga, G. L. LeMee and I. James. 'A frenzy of well drilling by California farmers leaves taps running dry'. *Los Angeles Times*. [[https://www.latimes.com/projects/california-farms-water-wells-drought/]{.underline}](https://www.latimes.com/projects/california-farms-water-wells-drought/). 16th December 2021.
3.  I. James. 'Despite California groundwater law, aquifers keep droppin in a 'race to the bottom for agricultural wells''. *Los Angeles Times*. [[https://www.latimes.com/environment/story/2021-12-16/its-a-race-to-the-bottom-for-agricultural-wells]{.underline}](https://www.latimes.com/environment/story/2021-12-16/its-a-race-to-the-bottom-for-agricultural-wells). 16th December 2021.
4.  R Pauloo. 'An Exploratory Data Analysis of California's Well Completion Reports'. [[https://richpauloo.github.io/oswcr_1.html]{.underline}](https://richpauloo.github.io/oswcr_1.html). 30th April 2018.
5.  A. Fulton, T. Dudley, and K. Staton. 'Groundwater Level Monitoring: What is it? How is it done? Why do it?'. [[https://www.countyofcolusa.org/DocumentCenter/View/4260/Series1Article4-GroundwaterLevelMonitoring]{.underline}](https://www.countyofcolusa.org/DocumentCenter/View/4260/Series1Article4-GroundwaterLevelMonitoring). Last Accessed 15th October 2022.
6.  J. Miles, 'Getting the Most out of scikit-learn Pipelines'. Towards Data Science. [[https://towardsdatascience.com/getting-the-most-out-of-scikit-learn-pipelines-c2afc4410f1a]{.underline}](https://towardsdatascience.com/getting-the-most-out-of-scikit-learn-pipelines-c2afc4410f1a). 29th July 2021
7.  'Typical water well construction and terms'. Ground Water Information Center Online. [[https://mbmggwic.mtech.edu/sqlserver/v11/help/welldesign.asp]{.underline}](https://mbmggwic.mtech.edu/sqlserver/v11/help/welldesign.asp). Last Accessed 15th October 2022.
8.  'Groundwater'. United States Geological Survey. [[https://www.usgs.gov/special-topics/water-science-school/science/groundwater]{.underline}](https://www.usgs.gov/special-topics/water-science-school/science/groundwater). Last Accessed 15th October 2022.
9.  CNN-LSTM-Based Models for Multiple Parallel Input and Multi-Step Forecast. https://towardsdatascience.com/cnn-lstm-based-models-for-multiple-parallel-input-and-multi-step-forecast-6fe2172f7668. Last accessed 17th November 2021.
10. A. Nielsen. 'Practical Time Series Analysis'. O'Reilly Media. ISBN: 9781492041658. October 2019
11. A. L. D'Agostino. 'Bounding Boxes for All US States'. [[https://pathindependence.wordpress.com/2018/11/23/bounding-boxes-for-all-us-states/]{.underline}](https://pathindependence.wordpress.com/2018/11/23/bounding-boxes-for-all-us-states/).23rd November 2018. Last Accessed 15th October 2022.
12. S. M. Lundberg S. Lee. 'A unified approach to Interpreting Model Predictions'. *NIPS\'17: Proceedings of the 31st International Conference on Neural Information Processing Systems*. [[https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf]{.underline}](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). 25th May 2017.
13. J. Brownlee. 'Prediction Intervals for Machine Learning'. *Machine Learning Mastery*. [[https://machinelearningmastery.com/prediction-intervals-for-machine-learning/]{.underline}](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/).17th February 2021. Last Accessed 15th October 2022.
14. J. Brownlee. 'Machine Learning Mastery How to improve Neural Network Stability'. *Machine Learning Mastery*. [[https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/]{.underline}](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/). 4th February 2019. Last Accessed 15th October 2022.
15. A. M. MacEachren, A. Robinson, S. Hopper, S. Gardner, R. Murray, M. Gahegan & E. Hetzler. 'Visualizing Geospatial Information Uncertainty:What We Know and What We Need to Know'. *Cartography and Geographic Information Science*, 32:3, 139-160, DOI: 10.1559/1523040054738936. 2005
16. S. Kandi. 'Prediction Intervals in Forecasting: Quantile Loss Function'. *Medium*. [[https://medium.com/analytics-vidhya/prediction-intervals-in-forecasting-quantile-loss-function-18f72501586f]{.underline}](https://medium.com/analytics-vidhya/prediction-intervals-in-forecasting-quantile-loss-function-18f72501586f). 5th September 2019. Last Accessed 15th October 2022.
17. W. Q: Meeker, G. J. Hahn, L. A. Escobar'Statistical Intervals: A Guide for Practitioners and Researchers'. *Wiley* ISBN: 978-0-471-68717-7. April 2017.
18. B. F. Froeschke, L. B. Jones, B. Garman, 'Spatio-temporal Models of Juvenile and Adult Sheepshead (Archosargus probatocephalus) in Tampa Bay, Florida from 1996 to 2016'. *Gulf and Caribbean Research*, 31(1): 8 -- 17.\]([[https://doi.org/10.18785/gcr.3101.04]{.underline}](https://doi.org/10.18785/gcr.3101.04)). 2020.
19. 'sklearn.preprocessing.StandardScaler'. *Scikit Learn*.[[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). Last Accessed 15th October 2022.
    """)
