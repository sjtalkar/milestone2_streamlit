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
        "Unsupervised Learning",
        anchor="unsupervised_learning",
    )

    st.markdown("""---""")

    st.markdown("""
## Motivation

The objective of this unsupervised learning analysis is, through
clustering, to try to identify groups of Township-Ranges and their
characteristics. Ideally this analysis would help reveal the
characteristics of the Township-Ranges with sustainable water situation
and those with systemic water problems.

## Methodology

The dataset contains 8 years of data for each Township-Ranges. We
decided not to average the yearly data to have one clustering for each
Township-Ranges but perform the clustering year by year. This allows us
to see if there are a lot of year-to-year variations of Township-Ranges
clustering.

We used 3 different techniques to perform clustering: K-Means
,Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
and Hierarchical Clustering. For all of these techniques, the models
were trained with a grid-search approach to find the hyperparameters
giving the best clustering results.

To choose the best k value for the K-Means algorithm, we ran the K-Means
algorithm for different values of *k* between 2 and 16 and evaluate the
clustering results based ond the following metrics: 
* The Calinski-Harabasz score (the sum of the between-clusters distance to intra-cluster distance, higher values indicate better cluster compactness),
* The Davies-Bouldin score (the average similarity of each cluster, lower values indicate better separation between clusters),
* The Silhouette score (the divergence between clusters considering both intra and inter cluster distances),
* The Inertia (sum of squared distances) of samples to their closest cluster center
""")
    st.image(load_image("unsupervised-k-means-metrics.png"), use_column_width=True)
    st.markdown("""Based on the above plots of the various clustering metric scores we get
 * the Calinski-Harabasz score (indicating better cluster compactness) is best when __k=2__
 * the Davies-Bouldin score (indicating better separation between clusters) is best when __k=14__ but also good when __k=2__
 * although the Silhouette scores increases with k, it remains similar (between 0.1 and 0.18), very far from the optimum value of 1
 * the Inertia doesn't show any "elbow" in the plot to help deciding the best value of k
    
The primary challenge faced is the absence of a clear indication of good *k*
clustering value on any of the 4 evaluation metrics. Having a too high
number of clusters doesn't seem appropriate for this analysis. Since the
Davis-Boudin score seems to indicate a better clustering score
for *k*=2 amongst the lower number of clusters and it matches with the
motivations, we decided to use that value for the rest of the
unsupervised analysis. With that value of *k* defined, we then applied
the 3 chosen clustering techniques on the dataset and analyzed the
results.

## Clustering Results

The K-Means algorithm showed an almost consistent year-to-year
classification of the Township-Ranges with just 10 Township-Ranges showing some variations in their
clustering for some years. The DBSCAN algorithm results were not
conclusive at all regardless of the hyperparameters used. Almost all
Township-Ranges were assigned to one cluster while just a handful of
Township-Ranges mostly""")
    st.image(load_image("unsupervised-k-means-results.png"), use_column_width=True)
    st.markdown("""
located in the South-East border of the valley were assigned to
individual clusters. Finally the Hierarchical Clustering algorithm where
either almost identical or very similar to the K-Means results, almost
identical to the DBSCAN results or, in just 1 case, proposing a very
different clustering.

## Analyzing the Clusters

Although we analyzed the results of that single different clustering
computed by one of the Hierarchical Clustering outputs, the analysis was
not very conclusive and we focused on the analysis of the results of the
K-Means clustering.

As can be clearly seen on the K-Means clustering map,
one of the cluster (cluster 1 shown in brown) is mainly located on the
South-East of the San Joaquin Valley, on a axis following the road 99
between the towns of Madera, Fresno, Tulare, Delan and Bakersfield in
the Madera, Fresno, Tulare and Kern counties.""")
    st.image(load_image("unsupervised-k-means-important-features.png"), use_column_width=True)
    st.markdown("""
As we analyzed the details of the most important features for the two
clusters, we found out that in addition to the South-East
location along the road 99:
* Township-Ranges in cluster 1 (in brown) have:
  * an average of their land surface covered for more than 65% by *Entisoils* of hydrologic group *B*. According to the [definition *Entisoils*](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/class/maps/?cid=nrcs142p2_053596) are soils with *little, if any horizon development*, *many are sandy and very shallow*. The hydrologic group *B* are soils that have *moderately low runoff potential when thoroughly wet. Water transmission through the soils is unimpended*. In summary sandy and shallow soils letting the water penetrate deeper.
  * higher depth from the ground surface to groundwater (GSE_GWE), meaning that it is required to dig deeper wells to reach groundwater. This is also illustrated by the wells having a deeper top and bottom perforation.
  * have a higher population density, which is normal as  we see that these Township-Ranges are mainly aligned along the road 99, between the towns between Madera and Bakersfield
  * cultivate more of crop *D12* which are almonds and known to be consuming a lot of water
  * have more non-native (i.e. planted) hardwood forest
* Township-Ranges in cluster 0 (in blue) have:
  * in general slightly richer soils with more water runoff potential when wet
    * [*Alfisols*](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/class/maps/?cid=nrcs142p2_053590) of hydrologic group *D*, are *soils that have an argillic, a kandic, or a natric horizon and a base saturation of 35% or greater*. The hydrologic group *D* being soils *with a high runoff potential when thoroughly wet*
    * [*Vertisoils*](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/class/maps/?cid=nrcs142p2_053611) of hydrologic group *D*, are *soils that have a high content of expanding clay*
    * [*Molisoils*](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/class/maps/?cid=nrcs142p2_053603) of hydrologic group *B* are *soils that have a dark colored surface horizon and are base rich*
  * water reservoirs with more capacity
  * higher average yearly precipitation

From the details above it seems that the cluster 1 (in brown) is mainly made of Township-Ranges
* located in the South-East along the road 99, between the towns between Madera and Bakersfield and thus with higher population density,
* have poorer sandy soils
* requires to dig deeper to reach groundwater
* have less precipitation
* have water reservoirs with less capacity

## Clustering Results vs Objectives
To try evaluate if this clustering result could help water resource management agencies to identify areas they should 
focus on, we used 
* the well depth, 
* the amount of wells drilled 
* the amount of well shortage reports

as potential proxies for measuring water resource sustainability. We compared the
Township-Ranges clusters with the top 30 Township-Ranges with the
deepest wells (biggest GSE_GWE value) drilled, the highest number of
wells drilled, the highest number of reported well shortages, averaged
over the 2014-2021 period.""")
    st.image(load_image("unsupervised-clusters-vs-wells.jpg"), use_column_width=True)
    st.markdown(""" In all three cases, the
clustering does not properly capture all of these Township-Ranges.

## Conclusion
In summary, although partially meaningful, the clustering fails at
achieving the objective of identifying Township-Ranges where water
management requires attention. Our hypothesis is that the features in
the dataset are not appropriate to perform such a clustering.    
    """)
