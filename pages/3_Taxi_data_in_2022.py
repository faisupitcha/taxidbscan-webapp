import folium.features
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
from geopandas.tools import sjoin
import geopandas as gpd
import pandas as pd
import altair as alt
import plotly.express as px
import geopandas
import requests
import folium
import json

###Set the streamlit page layout
st.set_page_config(
    page_title="Taxi data in 2022",
    page_icon="📊",
    layout="wide"
)

alt.themes.enable("dark")

# *******************************************************************************************************************

@st.cache_data
def make_choropleth(input_df, counties, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, geojson=counties, locations=input_id, color=input_column, locationmode="ISO-3",
                               color_continuous_scale=input_color_theme,
                               range_color=(0, max(input_df.num_points)),
                               scope="asia",
                               labels={'num_points':'Count'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth

# *******************************************************************************************************************

# load data

url = "https://github.com/pcrete/gsvloader-demo/raw/master/geojson/Bangkok-districts.geojson"
response = requests.get(url)
data = response.json()
states = geopandas.GeoDataFrame.from_features(data, crs="EPSG:4326")

with st.form(key='file_upload_form'):
    # เพิ่มปุ่มให้ผู้ใช้เลือกและอัพโหลดไฟล์
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'pdf'])

    # เพิ่มปุ่ม submit
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:

        # ถ้าผู้ใช้เลือกและอัพโหลดไฟล์ได้
        if uploaded_file is not None:
            # ตรวจสอบประเภทของไฟล์ที่อัพโหลด
            if uploaded_file.type == 'text/csv':
                # อ่านข้อมูล CSV และแสดงผล
                df = pd.read_csv(uploaded_file)
                taxi_data = df
            elif uploaded_file.type == 'text/plain':
                # อ่านข้อมูลจากไฟล์ข้อความ (txt) และแสดงผล
                text = uploaded_file.getvalue().decode("utf-8")
                st.write(text)
            elif uploaded_file.type == 'application/pdf':
                # แสดงข้อมูลว่าไฟล์เป็น PDF
                st.write("Uploaded PDF file.")
            else:
                # แสดงข้อความเมื่อไฟล์ไม่รองรับ
                st.write("Unsupported file type")

# *******************************************************************************************************************

            with st.sidebar:
                st.title('🚕 Taxi2022 Dashboard')
                
                day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                
                selected_day = st.selectbox('Select day', day_list, index=len(day_list)-1)
                df_selected_day = taxi_data[taxi_data['day_of_week'] == selected_day]

                color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
                selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

                st.write('Please press the submit button')

            # สร้าง GeoDataFrame จาก DataFrame ที่มีคอลัมน์ lat และ lon
            geometry = [Point(lon, lat) for lat, lon in zip(df_selected_day['startlat'], df_selected_day['startlon'])]
            df_selected_day_geo = gpd.GeoDataFrame(df_selected_day, geometry=geometry)
            # ทำ Spatial Join เพื่อเชื่อมต่อข้อมูลของจุดกับโพลีกอน
            joined = gpd.sjoin(df_selected_day_geo, states, how="inner", op="within")
            # นับจำนวนจุดในแต่ละโพลีกอนแล้วจัดกลุ่มรายการตามโพลีกอน
            num_points_by_state = joined.groupby('OBJECTID').size().rename("num_points")
            # เชื่อมต่อผลลัพธ์กับ GeoDataFrame ของโพลีกอน
            states_with_points = states.merge(num_points_by_state, on='OBJECTID', how='left')
            states_with_points_sorted = states_with_points.sort_values(by="num_points", ascending=False)
            # st.write(states_with_points_sorted.head())
            df_states_points = pd.DataFrame({
            'dname': states_with_points_sorted['dname'],
            'num_points': states_with_points_sorted['num_points']
            })
       
            # *******************************************************************************************************************

            # กำหนด page layout

            col = st.columns((4.5, 2), gap='medium')

            # *******************************************************************************************************************

            with col[0]:
                st.title('Total taxi pickups')
                choropleth = px.choropleth(states_with_points_sorted, geojson=states_with_points_sorted.geometry , locations=states_with_points_sorted.index, color='num_points',
                                        color_continuous_scale=selected_color_theme,
                                        range_color=(0, max(states_with_points_sorted.num_points)),
                                        scope="asia",
                                        labels={'num_points':'Count'},
                                        projection="mercator",
                                        hover_name=states_with_points_sorted.dname,  # ระบุชื่อเขตที่จะแสดงใน popup
                                        hover_data={'num_points': True}
                                        )
                choropleth.update_layout(
                    template='plotly',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=400,
                    width=800
                )
                choropleth.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(choropleth)
                
                st.markdown('#### Hourly Taxi Rides Visualization')
                hourly_count = df_selected_day.groupby('Hour').size()
                fig = px.bar(x=hourly_count.index, y=hourly_count.values)
                fig.update_layout(title='Number of Taxi Rides per Hour',
                                xaxis_title='Hour',
                                yaxis_title='Number of Taxi Rides')
                st.plotly_chart(fig)

            # *******************************************************************************************************************

            with col[1]:
                st.markdown('#### Top districts with the highest number of taxi pickups')

                # แสดงผลลัพธ์
                st.dataframe(df_states_points,
                            column_order=("dname", "num_points"),
                            hide_index=True,
                            width=None,
                            column_config={
                                "dname": st.column_config.TextColumn(
                                    "District",
                                ),
                                "num_points": st.column_config.ProgressColumn(
                                    "Count",
                                    format="%f",
                                    min_value=0,
                                    max_value=max(df_states_points.num_points),
                                )}
                            )
                
                average_time_in_trip = df_selected_day['timeintrip'].mean()
                average_Distance_in_trip = df_selected_day['dist'].mean()

                st.markdown('#### Customers spend on trips')
                st.metric(label="Time (Second)", value="{:,}".format(round(average_time_in_trip)))
                st.metric(label="Distance (Meter)", value="{:,}".format(round(average_Distance_in_trip)))
