import streamlit as st
import os
#import folium 
#from PIL import Image
#from streamlit_folium import folium_static
 
st.set_page_config(
    page_title="Thai taxi clustering",
    page_icon="🚕",
)

# # Directory path
# directory = '.'

# # List all files and folders in the directory
# contents = os.listdir(directory)

# # Print file and folder names
# for item in contents:
#     item_path = os.path.join(directory, item)
#     if os.path.isdir(item_path):
#         print(f"Folder: {item}")
#     elif item.endswith(".gz"):
#         print(f"File: {item_path}")

st.write("# TAXI CLUSTERING USING DBSCAN IN THAILAND, 2022 🔍🚕") 

# image = Image.open("vecteezy_taxi-png-grafico-clipart-projeto_21594382.png")
# st.image(image, width=700)

st.markdown(
    """Welcome to the web page displaying the results of the analysis and comparison of taxi passenger pick-up densities and distributions during different rush hours in Bangkok!

Traffic in Bangkok is complex and varies over time as taxis navigate to various destinations in the city. Understanding the trends and patterns of taxi travel is essential for smoother and more efficient traffic flow.

On this webpage, we have analyzed and compared the density and distribution of taxi passenger pick-up points during different rush hours to help you understand the traffic trends and conditions during each period of the day.

The results obtained will show areas with the highest concentration of taxi pickups and the distribution of passenger pick-up points, impacting movement and traffic conditions in those areas.

We hope that the data and results we present will be useful for planning your future travel efficiently. Thank you for visiting and using our website!

source -  https://www.iticfoundation.org/"""
)
