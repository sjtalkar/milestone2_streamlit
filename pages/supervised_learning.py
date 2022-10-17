# Import necessary libraries

import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
from streamlit_folium import folium_static 

sys.path.append("..")

from lib.multiapp import MultiPage
from lib.page_functions import load_image, set_png_as_page_bg
from lib.prediction_visualization import get_geo_prediction_df, draw_predictions_for_model, get_evaluation_error_metrics


DataFolder = Path("./assets/")
# Load the model once created
# model = pickle.load(open("filename", "rb"))

# Show the header image
# Reference : https://www.youtube.com/watch?v=vwFR2bXXzTw
# Reference for loading model:https://www.youtube.com/watch?v=M1uyH-DzjGE



def app():

    """
        This function in this page is run when the page is invoked in the sidebar
    """
    y_pred_df = None
    project_image = load_image("page_header.jpg")
    failure_spikes_image = load_image("failure_spikes.png")
    prediction_quantiles_image = load_image("prediction_quantiles_formula.png")
    prediction_uncertainty_image = load_image("prediction_uncertainty.jpg")
    ml_classic_target_prediction_image  = load_image("ml_classic_target_prediction.jpg")
    ml_shifted_target_prediction_image  = load_image("ml_shifted_target_prediction.jpg")
    ml_shifted_target_image  = load_image("ml_shifted_target.jpg")
    df_correlation_image  = load_image("df_correlation.png")
    feature_target_correlation_image = load_image("feature_target_correlation.png")
    scree_plot_image  = load_image("scree_plot.png")
    biplot_image = load_image("biplot.png")
    supervised_results_image = load_image("supervised_results.png")
    
    
    st.image(project_image, use_column_width=True)

    # Project Proposal
    ###########################################
    st.subheader("Supervised Learning", anchor="supervised_learning")
           
    y_pred_df = get_geo_prediction_df()
    counties_list = list(y_pred_df.COUNTY.unique())
    model_list  = list(y_pred_df.columns[-6:])
           
    county_selected  = st.selectbox(
                        'Predict for county:',
                         ['All'] + counties_list)
    model_selected  = st.selectbox(
                        'Predict from model:',
                         model_list)
    
    new_y_pred_df = y_pred_df.copy()
    if county_selected != 'All':
        new_y_pred_df[model_selected] = np.where(new_y_pred_df['COUNTY'] != county_selected, 0, new_y_pred_df[model_selected])

    
    if st.button("Predict"):
            st.subheader(f"Groundwater Depth Predictions From {model_selected}")
            st.caption(f"Township-Ranges in San Joaquin river basin")
            st.caption(f"County: {county_selected}")
            st.caption(f"The prediction values are not absolute. The error metrics for each of the models is tabulated below.")
            st.markdown("""---""")
            folium_static(new_y_pred_df.explore(column=model_selected, cmap='twilight_r'))
            st.markdown("""---""")
            #Streamlit cannot deal with geometry column
            new_df = new_y_pred_df.drop(columns=['geometry'])
            if county_selected == 'All':
                st.dataframe(new_df[['COUNTY', 'TOWNSHIP_RANGE', 'YEAR', model_selected]])
            else:    
                st.dataframe(new_df.loc[new_y_pred_df['COUNTY'] == county_selected][['COUNTY', 'TOWNSHIP_RANGE', 'YEAR', model_selected]])
        
    st.markdown("""---""")
    st.subheader("Error metrics: Evaluation of model on test set")
    error_eval_df = get_evaluation_error_metrics()
    st.dataframe(error_eval_df)  
    st.markdown("""---""")
    st.markdown("""
    ## Failure analysis 

An in depth analysis was conducted on failure on two counts. There were
certain townships for which quite curiously, there was a spike in depth
predicted by all models.
""")
    
    col1, col2 = st.columns([2, 1]) 
    with col1:
        st.caption("Failure analysis across models")
        st.image(failure_spikes_image, use_column_width=True)
    
    with col2:
        st.markdown("""
        As seen in adjoining figure, these spikes are between
target value (depth shifted) of 0-50 feet and then again between 150-200
feet.
""")
        st.markdown("""
Isolating these values, shows us that these two township ranges have a
very different groundwater depth in the test set (year=2020) from the
mean for all years past. Since current depth is the biggest predictor of
the target, this is a data issue that will need investigation in further
iterations of this study. The township ranges isolated for the spikes
(T10S R21E and T15S R10E) have a mean groundwater depth value of 15.18
and 248.85 feet respectively prior to 2020. The depth unaccountably
changes to 267.65 and 483.50 feet in 2020, indicating data issues.
""")
    
    
    st.markdown("""---""")
   
    st.markdown("""
    ## Prediction Uncertainty

The predictions made by the RandomForestRegressor are point estimates
for each of the Township-Ranges. We create a prediction interval to
demonstrate the uncertainty of prediction. “A prediction interval for a
single future observation is an interval that will, with a specified
degree of confidence, contain a future randomly selected observation
from a distribution”<sup>\[16\]</sup>. Linear regression estimates the
conditional mean of the response variable given certain values of the
predictor variables, while quantile regression aims at estimating the
conditional quantiles of the response variable. By combining the
predictions of two quantile regressors, it is possible to build an
interval. Each model estimates one of the limits of the interval. For
example, the models obtained for Q=0.1 and Q=0.9 produce an 80%
prediction interval (90% - 10% = 80%). This is the interval in which a
median point estimate will lie 80% of the time.

""", unsafe_allow_html=True)
    col1, col2 = st.columns([ 1, 2])
    with col1:
        st.caption("Prediction Quantiles Formula")
        st.image(prediction_quantiles_image, use_column_width=True)
    with col2:
        st.markdown("""
        In the adjoining quantile loss equation, with q having a value between 0 and
1, when yip \> yi (over-prediction) the first term will influence the
loss function and the second term will similarly influence it for under
predictions. So as q is set closer to 1, over-predictions will be
penalized more than under predictions. Regression based on quantile loss
provides sensible prediction intervals even for residuals with
non-constant variance and non-normal distributions.
        """)
        
    col1, col2= st.columns([1, 2])
    with col1:
            st.caption("Random Forest Prediction Quantiles")
            st.image(prediction_uncertainty_image, use_column_width=True)
           
    with col2:
            st.markdown("""Adjacent figure shows prediction values in each quantile
for the RandomForestRegressor. 
For instance, in the interactive map in the notebook, T16S 13E in Fresno
County indicated a lower quantile prediction of 415.75, median quantile
prediction of 462.11 and upper quantile prediction of 538.34. The
prediction quantile of 90% interval implies that there is a 90%
likelihood that the true outcome is in the 415.75 to 462.22 range.

While there is a range of values for every township-range, we do see
areas of sustainable depths in Stanislaus and San Joaquin counties in
the north of the San Joaquin valley, whereas lower Kern county towards
the southern tip of the basin is clearly at risk of much higher
groundwater depths.

    """)
    st.markdown("""---""")

    st.markdown("""
    ## Supervised Learning Motivation 

In predicting groundwater depth in the township ranges the primary goal
was to identify areas where the water resource management and affiliated
agencies should focus their effort to address water shortage and wells
over drilling. The second objective was to identify the most important
subset of factors contributing towards the predictions.

## Train-Test and Target Split

For the machine learning approach converting the multivariate, multi-indexed (Township-Range + year) dataset to a supervised learning prediction task required (Appendix 4):
- Including the current groundwater depth as a feature : (GSE_GWE renamed as CURRENT_DEPTH)
- Shifting the groundwater depth of the next year as the prediction target for the current year.
The shift resulted in the loss of one feature data in 2021. Which implied that while each train, test and prediction set contain all Township-Ranges, the train set years are 2014-2019, the test set year is 2020 and the prediction set included year 2021. While transformation of the target is typically not necessary, in the context of deep learning, “A target variable with a large spread of values, in turn, may result in large error gradient values causing weight values to change dramatically, making the learning process unstable.”[13] Sklearn’s TransformedTargetRegressor was used to wrap the regressors used so that the transformation can occur in the pipeline without manually converting the target. Our target depth varied from 0.5 to 727 feet and we observed improvement in metrics upon transformation.
For the deep learning approach, to fit our dataset and objective, as well as Long Short-Term Memory (LSTM) neural network architectures, we split the train-test sets by group (Township-Ranges) and the inputs and targets by time (refer to Appendix 7.1 for the diagram). We used 15% of the Township-Ranges for the test set. This means that models were trained on 406 Township-Ranges and tested on 72 of them. During training 10% of the training data was used for cross-validation.
""")
   
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.caption("Classic ML")
        st.image(ml_classic_target_prediction_image, use_column_width=True)             
    with col2:
        st.caption("Convert timeseries to supervised learning")
        st.image(ml_shifted_target_prediction_image, use_column_width=True) 
    with col3:
        st.markdown(
            """
            The method used to capture part of the intrinsic time series nature of the data, the target variable is preserved as an input and the target becomes the target variable itself shifted by one time stamp.
Instead of predicting Y(t) based on on the features X1(t) - X4(t), the output Y(t) is then predicted based on the previous value of the features (X1(t-1) - X4(t-1)) but also based on it’s own previous value Y(t-1).
            """)
 
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Supervised Learning dataset")
        st.image(ml_shifted_target_image, use_column_width=True) 


    st.markdown("""
    ## The Machine Learning Approach

### Feature Subset from Feature Correlation and Dimensionality Reduction
    """)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Feature correlation")
        st.image(df_correlation_image, use_column_width=True) 
    with col2:
        st.markdown(
            """
            The dataset contains 80 features, some of which are intuitively highly
correlated (\>0.50) such as well yield and its static water level and
dimensions. Average precipitation is correlated to reservoir capacity by
a Pearson correlation value of 0.31. Crops such as Onion and Garlic are
correlated to Tomatoes. And [Blue oak-gray pine which](https://chaparrallmi.weebly.com/blue-oak.html) forms
one of the largest ancient forest types in California is naturally
correlated to Chaparral with which it mingles at lower elevation. To
select features using quantitative measures, feature correlations to
each other and to target were studied along with top features
constituting the principal components of a PCA analysis.
        """)
        st.markdown("""Unsurprisingly, predicted depth correlates strongly with current depth.
Correlation with other features is significantly weaker (\<0.3). Well
depth, precipitation, and crops such as Onion and Garlic along with
pasture land crops are more correlated with the target than others.
(Appendix 5). Some other correlations that were noticed to be higher
than others in the same category are those between Onions and Garlic
crops that are tolerant to arid soils, mixed pastures to precipitation
and current depth to ground surface
elevation.
     """)
    col1, col2= st.columns([1, 2])
    with col1:
            st.caption("Scree plot")
            st.image(scree_plot_image, use_column_width=True)
           
    with col2:
            st.markdown("""
            After observing the feature correlation in the heatmap, we decided to
check for latent features in the dataset that will effectively combine
the correlated features, and create a new set of features that are a
weighted linear combination of original features, using PCA. The number
of components selected explain 75% (a threshold we selected based on the
Scree plot), of the original variance of the dataset. The PCA
biplot below, emphasizes the collinearity we saw in the heatmap in
features related to a well such as completed depth, top and bottom of
perforations (for filter). Vectors with a smaller angle of separation
are more correlated.
            """)
            
    col1, col2= st.columns([3, 1])
    with col1:
            st.caption("Biplot with first two components")
            st.image(biplot_image, use_column_width=True)

            st.markdown("""
- The first component contains well features in an area such as its depth and static water level.
- The second component includes well counts and ground surface elevation.
- The third component is largely about precipitation and reservoir capacity, and well counts.
- The fourth component includes precipitation, ground surface elevation and population density and well counts.

And so we derive the sense that well counts play a significant role in
the variance of the data along with ground surface elevation, well
features, precipitation and reservoir capacity.""")

    st.markdown("""
    Feature selection was also narrowed by iteratively applying ML algorithms and studying feature 
    importance indicated by individual algorithms as well as model agnostic SHAP predictions.
    For instance, a summary plot for Random Forest with the entire train set as background distribution
    shows current depth being the highest predictor of the future depth target, followed by arid soils
    which increases depth prediction. Increase in precipitation decreases depth and interestingly decrease
    in population, perhaps because in urban areas there is more dependence on reservoir supply than
    groundwater supply. (Fig. 6) Note that these cannot be interpreted as being causal in nature.

### Setting a baseline through dummy regressor and linear regression

Machine Learning algorithms were applied on the dataset only after
setting a baseline to account for complexity-accuracy tradeoff and to
determine if the more complex models meet the required deployment
performance. A naive assumption can be made about the relation between
the features and the target initially and so before using complicated
models, running a Dummy Regressor or Linear Regressor sets a score that,
if improved upon, determines viability of the complex models.

Since this is a regression problem with continuous variables, the
evaluation scores find the difference (error) between the predicted
value and the observed value. From among the choices of R-squared, Mean
Square Error, Root Mean Square Error, we selected the mean magnitude of
the errors in a set of predictions (Mean Absolute Error) as the error to
be communicated since it uses the same units as the target and is
relatively easy to understand in the context. The Mean Absolute Error
(MAE) is calculated by taking the summation of the absolute difference
between the actual and predicted values of each observation over the
entire set and then dividing the sum obtained by the number of
observations in the dataset. These are negatively oriented scores which
means lower scores are better.

### **R-squared**

For the initial comparison of algorithms, before hyper tuning and for
generating a list of models for PyCaret to compare, R-squared gave us a
quick indication of the relative performance. R2 provides the proportion
of the variance for the target that's explained by selected features in
the model. An R-Squared value of 0.9, for instance, would indicate that
90% of the variance of the dependent variable being studied is explained
by the variance of the independent variable.

It is independent of the scale of the features and ranges from 0 to 1. A
negative value implies worse than the mean model. When evaluating an
algorithm, it is prudent to look into multiple regression scores and not
just R-squared since the acceptable threshold of the error will depend
on the distribution of the target value itself. To keep the groundwater
depth prediction difference from the mean in check we additionally
looked into MAE, MSE and RMSE.

### Reasoning behind the models selected

The baseline R-squared, established with DummyRegressor is 0.054 is a
very low value. PyCaret's compare_model initial results which hinted at
tree based models being among the top five including
GradientBoostedTrees (creates sequential learners with learning rates
that works very well with heterogeneous tabular data with categories
mixed in), RandomForestRegressor (ensemble bagging model) and
CatBoostRegressor (which is surprising since the dataset has continuous
variables) and XGBoostRegressor. PyCaret also did not pull up Support
Vector Machines in the first five top models although testing this
algorithm with a radial based kernel showed early promise with the many
featured dataset. Other advantages of SVM considered:

-   Effective in high dimensional spaces and where number of dimensions > number of samples.
-   Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
-   Versatile: different [Kernel functions](https://scikit-learn.org/stable/modules/svm.html#svm-kernels) can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.[SKlearn documentation](https://scikit-learn.org/stable/modules/svm.html)
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Supervised Learning Evaluation")
        st.image(supervised_results_image, use_column_width=True)
            
    with col2:
        st.markdown("""Manually setting up the parameter grid for RandomForestRegressor and SVR
after this initial analysis through PyCaret, provided better results in
optimizing the parameters through RandomSearchCV. R-squared shows how
well the data fit the regression model.""")

        st.markdown("""On evaluation on both train and test set results and a combination of
evaluation metrics,R-squared and MAE, RandomForestRegressor has the
better train score and acceptable generalizability test scores of the
various models. Advantages of Random Forest algorithm over Decision
Trees and SVM :
-   It can be understood easily
-   It can handle large datasets efficiently
-   Provides more regularization over decision trees.""")

    st.markdown("""The other factors considered were number of samples and absence of
categorical features and sensitivity to outliers. Results of the top
three algorithms are shown in Table X. Cross-validation for hypertuning
of parameters of each of the models, was carried out using
[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
which unlike GridSearchCV, does not try out all possible parameter
values, but rather a fixed number of parameter settings, sampled from
the specified distributions [Sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

""")
    
