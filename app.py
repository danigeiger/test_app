import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title('Iris model inference')
st.subheader('This app is designed to inference the iris model')

with st.sidebar:
    st.header('Data Requirements')
    st.caption('To inference the model you need to upload a dataframe in csv format with four columns/features(columns names are not important')
    with st.expander('Data format'):
        st.markdown(' - utf-8')
        st.markdown(' - separeated by coma')
        st.markdown(' - delimited by "."')
        st.markdown(' - first row - header')
    st.divider()
    #st.caption('Developed by me')
    #Use html to perfect your web interface the way you want it.
    st.caption("<p style = 'text-align:center'>Developed by me</p>", unsafe_allow_html=True)    

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click = clicked, args = [1] )


st.title("Embedded YouTube Video in Streamlit")

# Replace with your actual YouTube video ID
video_id = "1bOuNIMH0hE"

# Embed the video using markdown (iframe)
st.markdown(
    f'<iframe width="700" height="400" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>',
    unsafe_allow_html=True
)

video_id2 = "QSTwIo9IQRI"

# Embed the video using markdown (iframe)
st.markdown(
    f'<iframe width="700" height="400" src="https://www.youtube.com/embed/{video_id2}" frameborder="0" allowfullscreen></iframe>',
    unsafe_allow_html=True
)

import streamlit as st

st.title("Displaying an Image in Streamlit")

# Load an image from the local directory
st.image("CRLR_RAMP1.jpg", caption="This is a sample image", use_container_width =True)



if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader('Choose a file', type = 'csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory = True)
        st.header('Uploaded data sample')
        st.write(df.head())
        model = joblib.load('model.joblib')
        pred = model.predict_proba(df)
        pred = pd.DataFrame(pred, columns = ['setosa_probability', 'versicolor_probability', 'virginica_probability'])
        st.header('Predicted values')
        st.write(pred.head())
        pred = pred.to_csv(index=False).encode('utf-8')
        st.download_button('Download prediction', 
                        pred,
                        'prediction.csv',
                        'text/csv',
                        key = 'download-csv'
    )
    
