import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from copy import deepcopy

plt.rcParams["figure.figsize"] = (9,2)

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


header = st.container()
dataset = st.container()
#section = st.container()
my_expander1 = st.expander(label = "Histograms")
my_expander2 = st.expander(label = "Line plots")


@st.cache
def load_data_csv(file_byte_object):
    return pd.read_csv(file_byte_object, header=0, delimiter=',')


@st.cache
def load_data_xls(file_byte_object):
    return pd.read_excel(file_byte_object)


with header:
    # Title of app
    st.title("Streamlit app")
    #st.text("This is a test application")

# Add a sidebar
st.sidebar.subheader("Menu")

with st.sidebar:
    # set up file upload:
    uploaded_file = st.file_uploader("Upload file:", type = ["csv", "xlsx"])    # returns byte object


with dataset:

    df, df_names = None, ()
    if uploaded_file is not None:

        try:
            df = load_data_csv(uploaded_file)
            df_names = tuple(df.columns)
            
        except Exception as ex:
            print(ex)
            df = load_data_xls(uploaded_file)
            df_names = tuple(df.columns)

        try:
            st.write("Imported data:")
            #st.write(df)
            #st.dataframe(df)
            #st.table(df)
        except Exception as e1:
            st.write("Please upload file to application!")            

        with st.sidebar:
            if st.checkbox('Select all'):
                selected_options = st.multiselect("Select one or more options:", list(df_names), list(df_names))
            else:
                selected_options =  st.multiselect("Select one or more options:", list(df_names))
            #st.write('You selected:', selected_options)

        # or on sidebar menu
        #selected_options = st.sidebar.selectbox(
        #   'Select a column:',
        #   ("All",) + df_names
        #   )

        # Display it:
        #st.write('You selected:', selected_options)
        #print(selected_options)

        # Select subset of columns
        if len(selected_options) > 0:
            filter_data = df[selected_options]
            #st.dataframe(filter_data)
        else:
            #st.dataframe(df)    
            filter_data = deepcopy(df)
        st.dataframe(filter_data)


        with my_expander1:
            st.subheader('Plot 1')
            for vals in selected_options:
                # Hisrogram setup
                fig, ax = plt.subplots()
                ax.hist(df[selected_options].values, bins=30, alpha=0.5, histtype='stepfilled')
                st.pyplot(fig)

        with my_expander2:
                st.subheader('Plot 2')
                st.line_chart(df[selected_options])        
            
    else:
        st.write("No dataset uploaded.")        


# Create dropdown in main box
#my_dropdown = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone'))

# or on sidebar menu
# my_dropdown = st.sidebar.selectbox(
#      'How would you like to be contacted?',
#      ('Email', 'Home phone', 'Mobile phone')
#      )

# # Display it:
# st.write('You selected:', my_dropdown)


# Add a slider to the sidebar:
# my_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

# st.write('You selected:', my_slider)

# buttons:
#https://blog.streamlit.io/introducing-submit-button-and-forms/

# with section:
#     st.title("my data")

#     fig = go.Figure(data=[go.Table(header = dict(values=['A Scores', 'B Scores']),
#                  cells = dict(values = [[100, 90, 80, 90], [95, 85, 75, 95]]))
#                      ])
#     fig.show()

# with my_expander2:
#     chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
#     st.line_chart(chart_data)