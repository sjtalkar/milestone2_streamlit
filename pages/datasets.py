# Import necessary libraries


from pathlib import Path
import streamlit as st
from PIL import Image

import sys

sys.path.append("..")

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
    st.subheader("Datasets", anchor="datasets")

    dataset_image = load_image("groundwater_modeling.png")
    voronoi_image = load_image("crop_voronoi.jpg")
    missing_values_image = load_image("missing_values.png")
    pipeline_image = load_image("sklearn_pipeline.jpg")
    plss_image = load_image("plss_info.png")

    st.image(dataset_image, use_column_width=True)

    st.markdown(
        """
        ## Data Sources

We have collected 10 geospatial datasets from federal and state (CA)
government agencies for the 2014-2021 period, on the factors impacting
groundwater depth:

San Joaquin Valley Public Land Survey System data, current groundwater
levels and consumption through well completion reports, agricultural
crops, population density, regional vegetation, reservoir capacity,
recharge through precipitation and soils survey (see Appendix 2).
Although we retrieved the data for dry wells and water shortage reports,
this dataset was not included as this is a voluntarily reported
information which is not verified, is not complete and has potential
errors.

 `Note`  Click on Dataset links for explanations on features and
extraction


| Dataset                                                                                                                                           | Retrieval Location                                                                                         | Retrieval Mode                            | Format                 | Records | Time Period      |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------|------------------------|---------|------------------|
| [Crops](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/crops.md)                                        | [California Natural Resources Agency](https://resources.ca.gov/)                                    | Download                                  | Geospatial datasets    | 1159979 | 2014, 2016, 2018 |
| [Groundwater](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/groundwater.md)                            | [CNRA Groundwater](https://data.cnra.ca.gov/dataset/periodic-groundwater-level-measurements)        | API                                       | CSV                    | 5064676 | 2014-2022        |
| [Population](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/population.md)                              | [US Census Bureau](https://www.census.gov/data/developers/data-sets/planning-database.2016.html)    | API                                       | JSON                   | 5465    | 2014-2020        |
| [Precipitation](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/precipitation.md)                        | [CDEC Precipitation](https://cdec.water.ca.gov/snow_rain.html)                                      | Web scraping                              | HTML                   | 1418    | 2013-2022        |
| [Reservoir](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/reservoir.md)                                | [CDEC Water](https://cdec.water.ca.gov/reservoir.html)                                              | Web scraping                              | HTML                   | 224     | 2018- 2022       |
| [Water Shortage](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/shortage.md)                            | [CNRA Well Water Shortage Reports](https://data.cnra.ca.gov/dataset/dry-well-reporting-system-data) | Download                                  | CSV                    | 4792    | 2015-2022        |
| [Soils](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/soils.md)                                        | [USDA Soils](https://nrcs.app.box.com/v/soils)                                                      | Download + manual extraction of the table | Microsoft Access Table | 2139    | 2016             |
| [Vegetation](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/vegetation.md)                              | [USDA Forest Service](https://data.fs.usda.gov/geodata/edw/datasets.php)                            | Download                                  | Zip File               | 54809   | 2018, 2019       |
| [Well Completion](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/well_completion.md)                    | [CNRA Well Completion Reports](https://data.cnra.ca.gov/dataset/well-completion-reports)            | API                                       | JSON                   | 2139    | 2014-2022        |
| [Census Bureau Shapefile](https://github.com/sjtalkar/milestone2_waterwells_deepnote/blob/master/doc/assets/plss_sanjoaquin_riverbasin.md) | [L.A. Times GitHub page](https://github.com/datadesk/groundwater-analysis/tree/main/data)           | Download                                  | GeoJSON                | 478     | Constant         |
"""
    )
    st.markdown("""---""")
    st.markdown(
        """
## Spatial Data Granularity 
#### PLSS Township-Range

According to Sidewell “The Public Land Survey System”, the “Public Land Survey System (PLSS)
is the surveying method developed and used in the United States to plat, or divide, land for sale and 
settling”. “Under this system the lands are divided into ‘townships,’ 6 miles square, which are related to 
base lines established by the federal government. The township numbers east or west of the Principal Meridians 
are designated as ranges; whereas, the numbers north and south of the base line are tiers”.
“Thus, the description of a township as ‘Township 16 North, Range 7 West’ would mean that the township is 
situated 16 tiers north of the base line for the Principal Meridian and 7 ranges west of that meridian. Guide Meridians, at intervals of 24 miles east and (or) west of the Principal Meridian, are extended north and (or) south from the base line; Standard Parallels, at 24-mile intervals north and (or) south of the base line, are extended east and (or) west from the Principal Meridian.
The Township is 6 miles square. It is divided into 36 square-mile “sections” of 640 acres, each which may be 
divided and subdivided as desired. The diagram below shows the system of numbering the sections and the usual 
method of subdividing them”.

`Credit:` “The Public Land Survey System (PLSS)”. Sidwell.
https://www.sidwellco.com/company/resources/public-land-survey-system/. Last Accessed
15th October 2022.

## Data Manipulation and Aggregation

All the datasets were aggregated at the Township-Range dimension.
For each dataset we computed each feature value per
Township-Range and year. To do so we performed two main types of data
transformation.

The first one was to overlay the Township-Range boundaries over
geospatial data like crops, soils, vegetation, population density to
either compute the land surface used by each feature in each Township
Range and year (e.g., the land surface used by each crop).

Some datasets, like precipitation, provide point measurements instead of
spatial ones. To transform such data into spatial data, we used the
Voronoï Diagram method. It computes the Voronoï polygons by using the
measurement points as the Voronoï polygons center. We then overlaid the
Township-Ranges boundaries on top of the computed Voronoï diagrams and
for each Township-Range took the mean value of all the intersecting
polygons."""
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.caption("Overlaying Township-Range borders on the Crops dataset")
        st.image(voronoi_image, use_column_width=True)

    with col2:
        st.markdown(
            """
The result of the transformations and aggregation applied was a dataset
of 478 Township-Ranges, each containing a multivariate (80 features)
time series (data between 2014 to 2021).
"""
        )

    st.markdown("""## Imputation Pipeline""")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Missing values")
        st.image(missing_values_image, use_column_width=True)

    st.markdown(
        """
As seen in above figure, there are missing data that need
to be addressed. We used a pipeline to both impute and normalize data
after the dataset is split into train and test sets. Creating this
pipeline and fitting it to the train set makes the application of
normalization uniform across the datasets when we then transform the
test set. It also guards against data leakage. While crops, soil and
vegetation were imputed with FunctionTransformers applying
forward-fills, for reservoir capacity (set as minimum of all years) and
ground surface elevation (set as median of all years data) we created
custom group transformers, grouped by township range.
"""
    )

    # col1, col2 = st.columns(2)
    # with col1:
    st.caption("Imputation Pipeline")
    st.image(pipeline_image, use_column_width=True)

    st.markdown(
        """
**What is lookahead:** The pipelines we created posed a special challenge in splitting the data since several of the imputations required past information from the same township range which were in the training set alone. We needed to guard against lookahead to avoid leaking knowledge of the future when designing, training, or evaluating a model. 

**Walk-forward validation:** In walk-forward validation, the dataset is first split into train and test sets by selecting a cut point, we chose this as the year 2019 e.g. data for years 2014-2019, 2020 is used for testing and 2021 for prediction. 

While tree based algorithms are not sensitive to scale of data and do
not require scaling and normalization, algorithms such as Support Vector
Regression (SVR) that aim at maximizing the margin between the
hyperplane and the closest supporting data vectors are sensitive to the
scale of individual features. Both of these are discriminative models,
predicting y given x. We evaluated MinMaxScaler (range 0-1) and
StandardScaler (subtracting mean and unit variance) found that
evaluation metrics and PCA components generated varied based on the
scaling used with StandardScaling resulted in a marginally better
R-squared and a more intuitive set of top features in PCA components. As
stated in [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
"many elements used in the objective function of a learning algorithm
(such as the RBF kernel of Support Vector Machines) assume that all
features are centered around 0 and have variance in the same order."

However, in order to have the values between 0 and 1 for the deep
learning algorithm, a min-max scaling method was used.
        """
    )
    st.markdown("""---""")

