import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import global_config as glob
from copy import deepcopy
import os, openai
from importlib import reload

#reload(glob)

#openai.api_key = os.getenv("OPENAI_API_KEY")  
openai.api_key = glob.UC_OPENAI_API_KEY

#plt.rcParams["figure.figsize"] = (9,2)

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

#---------------------------------------------------------
header = st.container()
dataset = st.container()
text_input = st.container()
my_expander1 = st.expander(label = "Histograms")
my_expander2 = st.expander(label = "Line plots")
#----------------------------------------------------------

@st.cache
def load_data_csv(file_byte_object):
    return pd.read_csv(file_byte_object, header=0, delimiter=',')


@st.cache
def load_data_xls(file_byte_object):
    return pd.read_excel(file_byte_object)


@st.cache
def create_filter_string(lookup : dict, lookup_assigned : dict, verbose : bool = False) -> str:
    query_string = ""
    for i, cols in enumerate(lookup.keys()):
        if len(lookup_assigned[cols]) > 0: 
            if i==0:
                query_string += "{} == {}".format(cols, lookup_assigned[cols])
            else:    
                query_string += ' & ' + "{} == {}".format(cols, lookup_assigned[cols])
        if verbose : print(query_string)
    return query_string 

@st.cache
def filter_data(df : pd.DataFrame, query_string : str):
    return df.query(query_string)


@st.cache
def make_line_plot(df : pd.DataFrame, used_columns = ['time', 'target2', 'est2', 'lower2', 'upper2']):
    """Create trace plot"""
    fig = go.Figure([
    go.Scatter(
        x = df['time'], y = df['target2'],
        line=dict(color='rgb(31, 119, 180)'),
        mode='markers+lines', name = 'Observed'
    ),
        go.Scatter(
        x = df['time'].tail(1), y = df['target2'].tail(1),
        mode='markers', name = 'Predicted out-of-sample', fillcolor='rgba(68, 68, 68, 0.3)'
    ),
        go.Scatter(
        x = df['time'], y = df['est2'],
        line=dict(color='rgba(80,50,50,0.5)'),
        mode='markers+lines', name = 'Predicted in-sample'
    ),
        go.Scatter(
        x = df['time'], y = df['upper2'],
        line=dict(color='rgba(50,50,50,0.1)', width=4, dash='dot'),
        fill=None, name = 'prediction lower/upper bound')

    , go.Scatter(
        x = df['time'], y = df['lower2'],
        line=dict(color='rgba(50,50,50,0.1)', width=4, dash='dot'),
        fill='tonexty', name = 'prediction lower bound', showlegend=False
    )
    ])

    fig.update_layout(
        yaxis_title='target',
        title='Target with 1-month ahead prediction', legend={'traceorder':'normal'}, 
        hovermode="x")

    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", tickangle = 90)    
    return fig


@st.cache
def make_hist_plot(df : pd.DataFrame, used_columns = ['time', 'target1', 'est1', 'lower1', 'upper1']):
    """Create histogram based on absolute frequencies"""
    counts, bins = np.histogram(df['target2'], bins=30)
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = px.bar(x=bins, y=counts, labels={'x':'target', 'y':'frequency'})
    return fig


##########################################################################################################################################
################ Start app
###################################

with header:
    # Title of app
    st.title("NLP playground")
    #st.text("This is a test application")

# Add a sidebar
st.sidebar.header("Menu")

with st.sidebar:
    # set up file upload:
    uploaded_file = st.file_uploader("Upload file:", type = ["csv", "xlsx"])    # returns byte object


#-------------------------------------------------------------------------
with text_input:
    
    txt = st.text_area(label = 'Enter text query here:', value="", height  = 400)
    
    if txt is not None:
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt=txt,     #"Create a SQL request to find all users who live in California and have over 1000 credits:",
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
        st.write('Input text:', txt)
        st.write('Generated text:', response['choices'][0]['text'])

#-------------------------------------------------------------------------------

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

        st.markdown("***")

        # Add a sidebar
        st.subheader("Plots:")

        with my_expander1:
            #st.subheader('Plot 1')
            #for vals in selected_options:
            #    # Histogram setup
            fig = make_hist_plot(df = filter_data)
            st.plotly_chart(fig)
        
            #fig, ax = plt.subplots()
            #ax.hist(filter_data['target2'].values, bins=50, alpha=0.5, histtype='stepfilled')
            #st.pyplot(fig)

        with my_expander2:
                #st.subheader('Plot 2')
                #st.line_chart(df[selected_options])       
                fig = make_line_plot(df = filter_data)
                st.plotly_chart(fig)     
            
    #else:
    #    st.write("No dataset uploaded.")        


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
