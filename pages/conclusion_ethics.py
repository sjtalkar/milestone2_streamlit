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
        "Ethical Considerations and Conclusion",
     
    )

    
    st.markdown(
        """
# Limitations

As the results and the failure analysis show, this project suffers
various limitations. The range of prediction uncertainty is too wide for
us to achieve the project’s objectives and advise our stakeholders with
enough confidence. For all approaches used, the metrics were too low to
promote the project production. More features and certainly more
historical data would have made the results more robust. Gathering
insights from environmental field experts would probably help us to
select more appropriate datasets, features but also a more suitable
prediction target to achieve our objectives.

# Conclusion

Despite the results not being sufficient to achieve the initial
objectives, this project was however a great learning opportunity. First
of all, we learned to manipulate and extract insights from geo-spatial
data. The multivariate time series format of the dataset also forced us
to go beyond classic Machine Learning techniques learned during the MADS
courses and learn and implement creative machine learning and deep
learning approaches. Although each approach used in the project
emphasized different features in regards to our objective, it did bring
some features like soil aridity and some crops to our attention and
would be useful findings in a second iteration of this project. The
analysis could have potentially suffered from confirmation bias with
emphasis on almond and pistachio crops in news articles. Correlations
and feature importance of the algorithms indicated that soil aridity,
and pasture land have higher impact on the predictions in our analysis
and could drive our next iteration.

**What we will do next**

This project will certainly benefit from a new iteration starting with a
qualitative analysis by including environmental experts in the team,
reaching out for more data and performing more feature engineering. A
better objective metric might also provide better prediction results.

# Ethical Considerations

We are keenly sensitive to the fact that the subject and set of stakeholders we have chosen to address has real world implications that could potentially impact the residents and farmers in the San Joaquin valley. By no means is this analysis complete or production ready and is a work that has many possibilities to be extended. It is meant to be thought-provoking in the features correlations to the target that it reveals. We have taken some precautions to communicate results effectively and forestall unintended usage of the prediction in this manner:

1.  We present results in terms of Township-Ranges in a county that lie
    within the San Joaquin valley as point estimates but clearly
    indicate the error of predictions in each model and also derive
    prediction quantiles for the best performing model.

2.  We highlight the fact that more historical data needs to be
    collected for the Deep Learning module to learn from the data
    meaningfully.

3.  We include the possibility of extending the search of features to
    include weather characteristics such as temperature, drought
    conditions and GDP of the area to be sensitive to the fairness in
    water distribution.

4.  We are considering replacing the groundwater depth in feet below
    ground surface (GSE_GWE) as a target feature by another feature
    which could better fit our objective of identifying areas with
    water resource issues.

5.  Ellul(1964) stated that technique and technical processes strive for
    the “mechanization of everything it encounters”. We have to remind ourselves at every turn
    in this analysis, that water forms the backbone of a human settlement
    and lives can be just as adversely affected by decisions made as a
    result of this analysis, as they can be improved. Our analysis is an
    attempt to employ learned as well as current techniques. As stated by
   Nielsen M.<sup>\[22\]</sup>,“Such improvements to the way discoveries
   are made are more important than any single discovery’”.


""", unsafe_allow_html=True)
    
    st.markdown("""---""")

