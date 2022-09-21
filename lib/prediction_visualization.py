import sys
sys.path.append('..')

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from lib.viz import display_data_on_map, simple_geodata_viz
from lib.township_range import TownshipRanges

def read_target_shifted_data(data_dir:str,
                             train_test_dict_file_name:str,
                             X_train_df_file_name:str,
                             X_test_df_file_name:str):
    """This function creates a dataframe to compare models based on mean absolute error,
        mean squared error. root mean squared error, and r-squared

    :param data_dir: directory containing pickled processed data files
    :param train_test_dict_file_name: file containing train and test numpy arrays
    :param X_train_df_file_name: file containing training feauture dataframe
    :param X_test_df_file_name : file containing testing feauture dataframe
    :return: retrieved dictionary of numpy arrays and dataframess
    """
    full_path = f"{data_dir}{train_test_dict_file_name}"
    with open(full_path, 'rb') as file:
            train_test_dict = pickle.load( file)
    
    full_path =  f"{data_dir}{X_train_df_file_name}"
    with open(full_path, 'rb') as file:
            X_train_impute_df = pickle.load( file)

    full_path =  f"{data_dir}{X_test_df_file_name}"
    with open(full_path, 'rb') as file:
            X_test_impute_df = pickle.load( file)
    return train_test_dict, X_train_impute_df, X_test_impute_df 



def get_predictions_df():
    """This function applies saved models on the prediction dataset and returns predictions for all TRs for prediction year 2021
     
        :param None
        :output: a  dataframe with columns for Year = 2021, TOWNSHIP_RANGE and columns for the predictions from each of the models
    """
    #county_tr_mapping.nunique() # 478 TRs repeated in counties
    data_dir = "../assets/train_test_target_shifted/"
    train_test_dict_file_name = "train_test_dict_target_shifted.pickle"
    X_train_df_file_name = "X_train_impute_target_shifted_df.pkl"
    X_test_df_file_name = "X_test_impute_target_shifted_df.pkl"

    train_test_dict, X_train_impute_df, X_test_impute_df = read_target_shifted_data(
        data_dir, train_test_dict_file_name, X_train_df_file_name, X_test_df_file_name
    )
    X_pred_impute = train_test_dict["X_pred_impute"]

    y_pred_df = train_test_dict['y_pred']
   
    with open('../assets/models/supervised_learning_models/models.pickle', 'rb') as file:
            models = pickle.load(file)
    for model in models:
        regressor_name = type(model.best_estimator_.regressor_).__name__ 
        y_pred_df[regressor_name] = model.best_estimator_.predict(X_pred_impute)
  
    y_pred_df.drop(columns=['GSE_GWE_SHIFTED'], inplace=True)
    y_pred_df.reset_index(inplace=True)
    return y_pred_df



def get_geo_prediction_df():
    """This function combines the predictions in the dataset for the year 2021 that was set aside with the geometry of
    the township ranges by county
     
    :param None
    :output: a  dataframe with columns for COUNTY, Year = 2021, TOWNSHIP_RANGE and columns for the predictions from
    each of the models
    """
    township_range = TownshipRanges()
    county_tr_mapping = township_range.counties_and_trs_df
   
    # The path of this file is relative to the supervised_learning.py page
    if os.path.exists("./pages/prediction_values.csv"):
        y_pred_df = pd.read_csv("./pages/prediction_values.csv", dtype={'YEAR':str, 'XGBRegressor': np.float64, 'SVR': np.float64,
                    'KNeighborsRegressor': np.float64, 'GradientBoostingRegressor': np.float64,
                    'CatBoostRegressor': np.float64})
    else:
        return pd.DataFrame()
    
    y_pred_df = county_tr_mapping.merge(y_pred_df, left_on='TOWNSHIP_RANGE', right_on='TOWNSHIP_RANGE') 
    return y_pred_df

def get_evaluation_error_metrics():
    """
    This function returns the error metrics from evaluating models on test set. The data was saved in 
    the file that will be read.

    :param : None
    :output : dataframe containing evaluation metrics
    """
    if os.path.exists('./pages/test_set_model_evaluation.csv'):
        return pd.read_csv('./pages/test_set_model_evaluation.csv')
    else:
        return pd.DataFrame()


def get_lstm_prediction_df(file_name: str = "./pages/lstm_predictions.csv"):
    """This function combines the LSTM predictions in the dataset for the year 2021 that was set aside with the geometry
    of the township ranges by county

    :param file_name: name of the file with the LSTM predictions
    :output: a  dataframe with columns for COUNTY, Year = 2021, TOWNSHIP_RANGE and columns for the predictions from
    each of the models
    """
    township_range = TownshipRanges()
    county_tr_mapping = township_range.counties_and_trs_df

    y_pred_df = pd.read_csv(file_name, dtype={'YEAR': str, 'GSE_GWE': np.float64})

    y_pred_df = county_tr_mapping.merge(y_pred_df, left_on='TOWNSHIP_RANGE', right_on='TOWNSHIP_RANGE')
    return y_pred_df


### Streamlit has an issue with handling geometry columns and so this function was not used and instead folium was used.
def draw_predictions_for_model(gdf:gpd.GeoDataFrame, model_name: str,county_name:str='All'):
    """
        This function takes in a county (or all counties) and will visualize predictions 
        User can select model for which the errors will be presented
        :param : gdf : Dataframe with counties, township ranges and model predictions
        :output: visualization of the predictions as an Altair chart
    """
    new_df = gdf.copy()
    if county_name != 'All':
        new_df[model_name] = np.where(new_df['COUNTY'] != county_name, 0, new_df[model_name])
    return simple_geodata_viz(new_df, feature= model_name, title='Prediction for county township ranges', year='2021',
                   color_scheme= 'blues',
                   draw_stations = False)