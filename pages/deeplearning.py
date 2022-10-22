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
from lib.prediction_visualization import get_lstm_prediction_df, draw_predictions_for_model


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
    st.image(project_image, use_column_width=True)

    # Project Proposal
    ###########################################
    st.subheader("Deep Learning", anchor="deeplearning")
           
    y_pred_df = get_lstm_prediction_df()
    counties_list = list(y_pred_df.COUNTY.unique())
           
    county_selected  = st.selectbox(
                        'Predict for county:',
                         ['All'] + counties_list)
    
    if "county_loaded" not in st.session_state:
        st.session_state.county_loaded = False
    
    new_y_pred_df = y_pred_df.copy()
    if county_selected != 'All':
        new_y_pred_df['GSE_GWE'] = np.where(new_y_pred_df['COUNTY'] != county_selected, 0, new_y_pred_df['GSE_GWE'])
    
    if st.button("Predict") or st.session_state.county_loaded:
            st.session_state.county_loaded = True
            st.subheader(f"Groundwater Depth Predictions From LSTM")
            st.caption(f"Township-Ranges in San Joaquin river basin")
            st.caption(f"County: {county_selected}")
            st.markdown("""---""")
            folium_static(new_y_pred_df.explore(column='GSE_GWE', cmap='twilight_r'))
            st.markdown("""---""")
            #Streamlit cannot deal with geometry column
            new_df = new_y_pred_df.drop(columns=['geometry'])
            st.dataframe(new_df.loc[new_y_pred_df['COUNTY'] == county_selected][['COUNTY', 'TOWNSHIP_RANGE', 'YEAR', 'GSE_GWE']])
        
    st.markdown("""---""")
    st.markdown("""
    ## Multiple Multivariate Time Series Predictions with LSTM
The dataset is made of 478 Township-Ranges, each containing a multivariate (80 features) time series (data between 2014 
to 2021). This dataset can thus be seen as a 3 dimensional dataset of
$478\ TownshipRanges\ *\ 8\ time stamps\ *\ 80\ features$
The objective is to predict the 2022 target value of `GSE_GWE` (Ground Surface Elevation to Groundwater Water Elevation - Depth to groundwater elevation in feet below ground surface) for each Township-Range.

LSTMs are used for time series and NLP because they are both sequential data and depend on previous states.
The future prediction *Y(t+n+1)* depends not only on the last state *X1(t+n), ..., Y(t+n)*, not only on past values of 
the feature *Y(t+1), ..., Y(t+n)*, but on the entire past states sequence.""")
    st.image(load_image("lstm_inputs_outputs.jpg"), use_column_width=True)
    st.markdown("""During training and predictions:
* Township-Ranges are passed into the model one by one
* each cell in the LSM neural network receives a Township-Range state for a specific year (the state of the Township-Range at a specific position in the series)
* each state (year) in the series is represented by a multidimensional vector of all 80 features (including the target feature Y `GSE_GWE`)

The output is the Township-Ranges next year's value for the specific feature Y `GSE_GWE`. The model is trained on 2014-2020 (7 years) data to predict 2021.
During inference the last 7 years (2015-2021) of data are passed as input to predict the 2022 value.""")
    st.image(load_image("lstm_table_to_cells.jpg"), use_column_width=True)
    st.markdown("""
## Preparing the Dataset
### The Train-Test Split
To fit our dataset and objective, as well as LSTM neural networks architecture we perform the train test split as follows:
* Training and Test sets will be split by Township-Ranges. I.e., some Township-Ranges will have all their 2014-2021 data points in the training set, some others will be in the test set.
* The model will be trained based on the 2014-2020 data for all features - including the target feature - with the training target being the 2021 value of the target feature.

With such a method, unlike a simple time series forecasting where the target feature is forecasted only based on its past value, we allow past value of other features (in our case cultivated crops, precipitations, population density, number of wells drilled) to influence the future value of the target feature.""")
    st.image(load_image("lstm-train-test-split.jpg"), use_column_width=True)
    st.markdown("""We do not create a validation dataset as we use Keras internal cross-validation mechanism to shuffle 
    the data points (i.e., the Township-Ranges) and keep some for the validation at each training epoch.
### Data Imputation and Scaling
Missing data imputation for a Township-Range is performed only using the existing data of that Township-Range (not the 
data of all Township-Ranges). For example:
* a *fill forward* approach is used for many fields like crops, vegetation and soils. The percentage of land use per crop in 2014 in a Township-Range is imputed into the missing year 2015 for that particular Township-Range.
* for fields like `PCT_OF_CAPACITY` (the capacity percentage of water reservoir), missing values in a Township-Range are filled using the min, mean, median or max values of that particular Township-Range. In this case the *fit* method of our custom impute pipeline does nothing. Not only the impute pipeline does not need to learn values from other Township-Ranges data points to impute missing values in a Township-Range, if it does it will not be able to impute data in the test set as we impute by Township-Range and the Township-Ranges in the test set are not seen when *fitting* the impute pipeline. The *transform* method, then simply fills the missing values in a Township-Range based on past values of that Township range. This way, we can split the train and test sets by Township-Range and impute missing value without any data leakage as the impute pipeline does not learn anything from the Township-Ranges in the test set.

We use a MinMax scaler to scale all values between 0 and 1 for the neural network, except for the vegetation, soils and 
crops datasets which are already scaled between 0 and 1.

It should be noted that we do not need to do any data imputation on the training and test sets *y* target feature since
 it does not have any missing data point.""")
    st.image(load_image("lstm_pipeline.jpg"), use_column_width=True)
    st.markdown("""## Training Different Models
We tried 3 different LSTM models:
* A simple model made of a single *LSTM* layer and an output *Dense* layer
* A model made of a *LSTM* layer followed by a *Dense* and *Dropout* layers before the output layer
* An Encoder-Decoder model made of 2 *LSTM* layers followed by a *Dense* and *Dropout* layers""")
    st.image(load_image("lstm_architectures.jpg"), use_column_width=True)
    st.markdown("""
Encoder-decoder architectures are more common for sequence-to-sequence learning e.g., when forecasting the next 3 days 
(output sequence of length 3) based on the past year data (input sequence of length 365). In our case we only predict 
data for 1 time step in the feature. The output sequence being of length 1 this architecture might seem superfluous 
but has been tested anyway. This architecture was inspired by the Encoder-Decoder architecture in this article: 
*[CNN-LSTM-Based Models for Multiple Parallel Input and Multi-Step Forecast](https://towardsdatascience.com/cnn-lstm-based-models-for-multiple-parallel-input-and-multi-step-forecast-6fe2172f7668)*<sup>[9]</sup>.

As such models are made for sequence to sequence learning and forecasting, the output of such a model is different from 
the previous ones. It has an output of size *[samples, forecasting sequence length, target features]*. In our case the 
forecasting sequence length and number of target features are both 1.
## Comparing the Different Models
| | MAE | MSE | RMSE |
|---------|--------|--------|--------|
| model 1 | 23.794 | 1208.830 | 34.768 |
| model 2 | 30.340 | 1814.699 | 42.599 |
| model 3 | 30.206 | 1610.085 | 40.126 |

Based on the model scores it turns out that the simplest of the three LSTM models is the one having the best scores.

However, considering all the measurements between 2014 and 2022, the `GSE_GWE` (Ground Surface Elevation to Groundwater 
Water Elevation - Depth to groundwater elevation in feet below ground surface) target value has a
* median of 137.09 (~41.7 meters)
* mean value of 167.37 feet (~50.9 meters)
* min value of 0.5 feet (0 meters)
* max value of 727.5 feet (221.6 meters)

A mean average error of 23.80 feet (7 meters), and root mean square error of 34.77 feet (10.4 meters) in the prediction 
is fairly large. Even the best model does not seem to be accurate enough to be useful.

### Training Data Size Tradeoffs and Sensitivity

To evaluate the impact of the amount of historical data used to train
the model and perform predictions, we recursively trained the model with
1 to 7 years of data. The first model was trained only on the training
set of 406 Township-Ranges 2020 data, to predict the 2021 groundwater
depth and tested on the 72 Township Ranges in the test set. Then the
model was trained with 2019-2020 data, then 2018-2020 and so on.""")
    st.image(load_image("lstm_data_sensitivity.png"), use_column_width=True)
    st.markdown("""We see
that although, at the beginning, the RMSE reduces as we
add yearly data to train the model, the improvement in prediction is
minor. The RMSE only reduces from \~42.5 feet to \~34 feet by increasing
the number of historical data from 1 year to 4 years. This indicates that just adding a little bit of yearly data to 
very little yearly data to train the neural network model is not enough to significantly improve its performance. 
As neural networks tend to require a lot of data, the hypothesis that having much more historical data would improve 
the model performance is still a valid hypothesis. But the analysis also suggests the hypothesis that the model 
performance issue might also be related to the quality of the data or the features used.

## Hyperparameters Sensitivity Analysis
We perform here an analysis of the best model's sensitivity to the following hyperparamters:
* the optimizer used (e.g. Adam RMSprop, Adagrad)
* the training validation datasets split
* the number of lstm units
* the learning rate
* the batch size
* the number of training epochs

To perform this analysis, we trained 33,345 LSTM models for all possible combinations (within the selected ranges of 
values) of those 6 hyperparameters on the best model, and recorded for each model, the Root Mean Square Error (RMSE) on 
the test set.

The below visualization displays for each hyperparameter, the concentration of models per RMSE score and the average 
RMSE mean (using the color) depending on the hyperparameter values (the lower and the more blue the peak is, the better 
the hyperparameter value is). This allows us to show if a specific hyperparameter tends to lead to lower or higher RMSE 
and to compare the distribution between two values of the same hyperparameters.""")
    st.image(load_image("lstm_hyperparameters_sensitivity.png"), use_column_width=True)
    st.markdown("""Looking at this visualization, we can see - with some surprise - that the hyperparameters which seem 
    to have the biggest impact on the model performance have little to do with the model architecture itself (the number 
    of LSTM units) but with how the model is trained.
* The choice of the optimizer seems to have the largest impact on the model performance, with both the mean and distribution of the RMSE for all models trained with an `Adagrad` optimizer being really bad.
* The training-validation percentage split seems to have little impact. The best performance is obtained with assigning 10% of the training data to the validation set, but with 15% of the training data to the validation set the results are close. We can also see that when assigning 5% of the data to the validation set, the distribution is much more even i.e., there are more models performing worse with a small validation set.
* The bigger the learning rate, the better the model performs in terms or RMSE. The distribution of all models RMSE shows that with a learning rate of 0.01, most models have low RMSE around 40. With a learning_rate of 0.001 we have a bimodal distribution of the RMSE with models performing either around 40 or very poorly around 150. With a learning rate of 0.0001, the distribution although still bimodal is more even with most models having an RMSE above 60.
* On the other hand, the smaller the batch size, the more models have a low RMSE.
* Although there is less of a difference if we compare similar values (e.g., 50 and 70 epochs or 270 and 290 epochs), we still see clearly that the bigger the number training epochs the more there are trained models with a low RMSE. With a low number of epochs there are more models have an RMSE around 150.
* The number of LSTM units, impacting the number of neurons in the LSTM model, seems to have less impact on the performance of the RMSE. The distribution of all models RMSE does show differences between 10 and 190 LSTM units but not as much as other hyperparameters. The strong bimodal distribution with a low number of LSTM units show that in this case models will either perform well or very poorly.

What is also interesting is that, as seen below, if we take the combination of all the best hyperparameters
* lstm_units: 60
* learning_rate: 0.01
* validation_split: 0.1
* batch_size: 32
* epochs: 290

we end up with an MAE of 29.38 and an RMSE of 39.8 both slightly worse (respectively 23.79 and 34.77) than the best 
model (model 1) we found previously. The best model is thus not necessarily obtained by the combination of the best 
individual hyperparameters.
                
| | MAE | MSE | RMSE |
|---------|--------|--------|--------|
| model 1 | 23.794 | 1208.830 | 34.768 |
| model 2 | 30.340 | 1814.699 | 42.599 |
| model 3 | 30.206 | 1610.085 | 40.126 |
| hyperparameters best combination | 29.383 | 1585.742 | 39.821 |
                
## Conclusion

The best model on the test set was the simplest model and the one we
used for the rest of the analysis. Once trained on the 2014-2020 data to
predict 2021, the model was used to predict the 2022 groundwater depth
based on the 2015-2021 data. The Township-Ranges with high or low
groundwater depth or with high or low year-to-year variation of
groundwater depth were retained by the model.""")
    st.image(load_image("lstm_2022_predictions_vs_past_data.jpg"), use_column_width=True)
    st.markdown("""But considering the
groundwater depth varies in the dataset from 0.5 to 727 feet and has a
mean of 167 feet, a root mean square error (RMSE) of 34.7 feet and mean
average error of 24.8 feet feels too high to achieve the objective of
predicting areas where water management requires attention.""")