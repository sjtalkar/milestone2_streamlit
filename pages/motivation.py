# Import necessary libraries

from pathlib import Path
import streamlit as st
from PIL import Image

import sys

sys.path.append("..")

from lib.multiapp import MultiPage
from lib.page_functions import load_image, set_png_as_page_bg


DataFolder = Path("./assets/")

# Show the header image


def app():

    """
        This function in this page is run when the page is invoked in the sidebar
    """

    project_image = load_image("project_motivation.png")
    st.image(project_image, use_column_width=True)
    st.markdown("""---""")

    # Project Proposal
    ###########################################

    st.markdown(
        """<p><strong>There's a whole fascinating world that exists underneath our feet that we don't see, therefore we don't relate.<br>--Erin Brokovich</strong></p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """# [A Thirsty Valley](https://www.latimes.com/projects/california-farms-water-wells-drought/)"""
    )
    st.markdown(
        """**Aim**: Predict San Joaquin Valley (CA) groundwater depth in feet"""
    )

    st.markdown(
        """    
California's Sustainable Groundwater Management Act
(SGMA)<sup>\[1\]</sup> was passed in 2014 with the intention to address
over pumping, halt chronic water-level declines and bring long-depleted
aquifers into balance. “Despite SGMA, a frenzy of well drilling has
continued on large farms across the **San Joaquin Valley, the state's
largest and most lucrative agricultural zone.** As a result, shallower
wells supplying nearly a thousand family homes have gone dry in recent
years.”<sup>\[2\]</sup>

Frequently, perniciously drought-inflicted California, depends on
groundwater for a major portion of its annual water supply, particularly
for agricultural and domestic usage. This project seeks to aid policy
makers and natural resource management agencies preemptively identify
areas prone to overdraft and bring groundwater basins into **balanced
levels of pumping and recharge.**

Focused on the San Joaquin Valley, the objectives are:

-   > **Supervised Learning:** Predict the groundwater depth in feet below
    > ground surface (GSE_GWE). This value portends shortage in a
    > TownshipRange. Increase or decrease in GSE_GWE will then indicate
    > if there will be more requests for well construction. This in turn
    > will provide a quantitative metric for whether SGMA is functioning
    > and areas to focus on for recharge.

-   > **Unsupervised Learning:** cluster areas into sustainable and
    > unsustainable areas. Detect groundwater depth anomalies within
    > Township Ranges in the river basin.

       
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """ 
        `Groundwater`
        
        Groundwater, which is found in aquifers below the surface of the earth, is one of our most important natural resources. 
        Groundwater provides drinking water for a large portion of California, nay, the nation's population. It also supplies 
        business and industries and is used extensively  for irrigation. California depends on groundwater for a major portion 
        of its annual water supply, particularly during times of drought. This reliance on groundwater has resulted in overdraft
        and unsustainable groundwater usage in many of California’s basins, particularly so in the San Joaquin River basin.
        
        `Sustainable Groundwater Management`
        
        Sustainable groundwater management is defined as managing water supplies in a way 
        that can be maintained without “causing undesirable results,” such as chronic declines in groundwater
        levels or “significant and unreasonable” depletion, adverse effects on surface water, degraded water
        quality or land subsidence.
        """
    )
    col1, col2 = st.columns([2, 1])
    groundwater_image = load_image("groundwater.png")
    with col1:
        st.image(groundwater_image, use_column_width=True, width=50)

    st.markdown(""" `Drought severity in California over the years`""")

    drought_image = load_image("drought_years_california.png")
    st.image(drought_image, use_column_width=True)

    st.markdown("""---""")

