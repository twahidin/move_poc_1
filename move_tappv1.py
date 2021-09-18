import streamlit as st #success this version does not support linear regression
# To make things easier later, we're also importing numpy and pandas for
# working with sample data. Need to upload this version to an ipad but we need to cut down on the face landmarks to save processing power
import numpy as np
import pandas as pd


st.title('MOVE Teacher Dashboard Prototype V1')

#access the database and send the data to google sheet




class_df = pd.DataFrame({
    'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
    'third column': ['1', '2', '3', '4', '5'],
    })


class_df = pd.DataFrame({
    'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
    'third column': ['1', '2', '3', '4', '5'],
    })

stream_df = pd.DataFrame({
  'str column': ['Express', 'Normal Academic', 'Normal Tech'],
  })


with st.sidebar.form("Class Analysis"):
    name = st.text_input('Class Code')
    age = st.slider('Age', min_value = 12, max_value = 17, value = 15, step=1)
    level = st.selectbox('Select your level:',class_df['sec column'])
    stream = st.selectbox('Select your stream:', stream_df['str column'])
    class_no = st.selectbox('Select your class:',class_df['third column'])
    submit_button = st.form_submit_button()