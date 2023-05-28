import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from lime import lime_tabular
import shap
from PIL import Image
#from alibi.explainers import CounterFactual

#Function Definitions

def EncodeHour(h):
    if(h<4):
        return 5
    elif(4<=h<8):
        return 1
    elif(8<=h<12):
        return 0
    elif(12<=h<16):
        return 4
    elif(16<=h<20):
        return 2
    elif(20<h):
        return 3

def Lime(X, instance, model):
    X_lime = np.array(X)
    explainer = lime_tabular.LimeTabularExplainer(X_lime, mode="regression", feature_names= X.columns.tolist())
    explanation = explainer.explain_instance(instance, model.predict, num_features=len(X_lime[0]))
    return explanation
    #explanation.show_in_notebook() 

def SHAP(X, instance, model):
    instance_array = instance.reshape(1, -1)
    model.predict(instance_array)
    k_explainer = shap.KernelExplainer(model.predict, X)
    shap_values = k_explainer.shap_values(instance)
    return shap_values, k_explainer.expected_value

#Function Definitions End

st.set_page_config(layout="wide")
st.title('Flight Price Prediction Web App')
st.write('This is a web app to predict the price of flights. Please adjust the value of each feature and click on the Predict button to see the predicted price.')

loaded_model = pickle.load(open('./flights_model.sav', 'rb'))
df = pd.read_csv("./DataSets/X_train.csv")

airline = 0
source_city = 0
destination_city = 0
stops = 0
departure_time = 0
arrival_time = 0
days_left = 0
duration = 0
flight_code = 0
flight_no = 0

with st.sidebar:

    #airline
    airline_mapping = {'Vistara':0,'Air_India':1,'GO_FIRST':2,'Indigo':3,'AirAsia':4,'SpiceJet':5}
    al = st.selectbox('Airline', ('Vistara','Air_India','GO_FIRST','Indigo','AirAsia','SpiceJet'))
    airline = airline_mapping[al]

    #flightCode
    flight_code_mapping = {'UK':0, 'AI':1, 'G8':2, 'E6':3, 'I5':4, 'SG':5}
    flight = st.selectbox('Flight', df['flight'].unique())
    flight_code = flight_code_mapping[flight.split("-")[0]]
    flight_no = int(flight.split("-")[1])

    #City From
    city_mapping={'Mumbai':0, 'Delhi':1, 'Bangalore':2, 'Kolkata':3, 'Hyderabad':4, 'Chennai':5}
    scity=st.selectbox('City From', ('Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'))
    source_city = city_mapping[scity]

    #City To
    dcity=st.selectbox('City To', ('Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'))
    destination_city = city_mapping[dcity]

    #Stops
    stops = st.radio('Stops', (0,1,2))

    #Departure & Arrival Date & Days Left
    depDate = st.date_input("Departure Date", datetime.date.today(), datetime.date.today())
    arrDate = st.date_input("Arrival Date", datetime.date.today(), datetime.date.today())
    days_left=(depDate-datetime.date.today()).days
    
    #Departure & Arrival Time & Duration
    depTime = st.time_input('Departure Time', datetime.time(12,00))
    arrTime = st.time_input('Arrival Time', datetime.time(12,00))
    arrival_time=EncodeHour(arrTime.hour)
    departure_time=EncodeHour(depTime.hour)
    dep_datetime = datetime.datetime.combine(datetime.date.today(), depTime)
    arr_datetime = datetime.datetime.combine(datetime.date.today(), arrTime)
    duration = (arr_datetime - dep_datetime).total_seconds() / 3660

features = {
  'airline':airline,
  'source_city':source_city,
  'departure_time':departure_time,
  'stops':stops,
  'arrival_time':arrival_time,
  'destination_city':destination_city,
  'duration':duration,
  'days_left':days_left,
  'flight_code':flight_code,
  'flight_no':flight_no
  }
  
features_df  = pd.DataFrame([features])

st.table(features_df)
prButton = st.button('Predict')
if prButton:    
    prediction = loaded_model['model'].predict(features_df)    
    st.write('The predicted price of flight is '+ str(int(prediction)))

    col1, col2 = st.columns((1,2))

    with col1:
        #lime
        lime_explanation = Lime(loaded_model['X'], features_df.values[0], loaded_model['model'])
        lime_fig = lime_explanation.as_pyplot_figure()
        lime_fig.subplots_adjust(left=0.5)
        image_stream = BytesIO()
        lime_fig.savefig(image_stream, format='png')
        plt.close(lime_fig)
        st.image(image_stream.getvalue())
    with col2: 
        #SHAP
        shap_values, expected_value = SHAP(loaded_model['X'], features_df.values[0], loaded_model['model'])
        shap_values_matrix = shap_values.reshape(1, -1)
        fig, ax = plt.subplots(dpi=80)
        shap.summary_plot(shap_values_matrix, features_df.values[0], plot_type="bar", show=False)
        ax.set_yticklabels(features_df.columns)
        fig.subplots_adjust(left=0.3)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80)
        plt.close()
        buffer.seek(0)
        st.image(buffer)


