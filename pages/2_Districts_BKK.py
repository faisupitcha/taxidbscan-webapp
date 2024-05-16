import folium.features
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import altair as alt
import geopandas
import requests
import folium
import json

###Set the streamlit page layout
st.set_page_config(
    page_title="Districts In Bangkok",
    page_icon="🗺",
    layout="wide"
)

alt.themes.enable("dark")

url = "https://github.com/pcrete/gsvloader-demo/raw/master/geojson/Bangkok-districts.geojson"
response = requests.get(url)
data = response.json()
states = geopandas.GeoDataFrame.from_features(data, crs="EPSG:4326")

# *******************************************************************************************************************

# กำหนด page layout

col = st.columns((4.5,4), gap='medium')

# *******************************************************************************************************************

with col[1]:
    # เลือกเขต
    st.markdown("#### Population")
    selected_district = st.selectbox('Select district', states['dname'])

    # นับจำนวนเพศแต่ละเขต
    selected_district_data = states[states['dname'] == selected_district]
    male_count = selected_district_data['no_male'].iloc[0]
    female_count = selected_district_data['no_female'].iloc[0]
    # นับจำนวนเพศรวมในเขต
    total_count = female_count+male_count

    # สร้าง DataFrame สำหรับ Pie Chart
    pie_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Count': [male_count, female_count]
    })

    # สร้าง Pie Chart ด้วย Plotly Express
    fig = px.pie(pie_data, values='Count', names='Gender', 
                labels={'Gender': 'Gender Distribution'}, 
                title='Gender Distribution in Selected District',
                hole=0.3,
                color_discrete_sequence=['blue', 'red'])  # กำหนดระยะห่างตรงกลางของ Pie Chart
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0),font=dict(size=20))
    # แสดง Pie Chart ใน Streamlit
    st.plotly_chart(fig, width=100, height=100)

    st.metric(label="Total population in district", value="{:,}".format(total_count))
    st.metric(label="Total population in Bangkok", value="5,494,932")

# *******************************************************************************************************************

with col[0]:

    st.title("DISTRICTS IN BANGKOK")

    map = folium.Map(location=[13.7563, 100.5018], tiles="OpenStreetMap", zoom_start=10)
    choropleth = folium.Choropleth(
        geo_data=data,
        data=states,
        columns=('dname', 'AREA'),
        key_on='feature.properties.dname',
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.8,
        highlight=True,
        legend_name="District Area"
    )
    choropleth.geojson.add_to(map)
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['dname'], labels=False)  # แก้เป็นชื่อคอลัมน์ที่ถูกต้อง
    )

    # หาขอบเขตของข้อมูลและกำหนดให้แผนที่ซูมเข้าไปที่ขอบเขตนั้น
    min_lat, min_lon = states.bounds.miny.min(), states.bounds.minx.min()
    max_lat, max_lon = states.bounds.maxy.max(), states.bounds.maxx.max()
    map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    st_map = st_folium(map, width=700, height=450)

    # แสดงผลลัพธ์
    df_states = pd.DataFrame({
    'District': states['dname'],
    'District_e': states['dname_e'],
    'Area': states['AREA'],
    'no_health': states['no_health'],
    'no_temple': states['no_temple'],
    'no_commu': states['no_commu'],
    'no_hos': states['no_hos'],
    'no_sch': states['no_sch']
    })

    st.dataframe(df_states,
                    column_order=("District", "District_e", "Area", "no_health", "no_temple", "no_commu", "no_hos", "no_sch"),
                    hide_index=True,
                    width=None,
                    column_config={
                    "District": st.column_config.TextColumn(
                        "District",
                    ),
                    "District_e": st.column_config.TextColumn(
                        "District_e",
                    ),
                    "Area": st.column_config.TextColumn(
                        "Area",
                    ),
                    "no_health": st.column_config.TextColumn(
                        "no_health",
                    ),
                    "no_temple": st.column_config.TextColumn(
                        "no_temple",
                    ),
                    "no_commu": st.column_config.TextColumn(
                        "no_commu",
                    ),
                    "no_hos": st.column_config.TextColumn(
                        "no_hos",
                    ),
                    "no_sch": st.column_config.TextColumn(
                        "no_sch",
                    )}
                    )
    
    st.markdown("source : https://github.com/pcrete/gsvloader-demo/raw/master/geojson/Bangkok-districts.geojson")
