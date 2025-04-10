import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import altair as alt #pip install altair
from plotly import graph_objs as go
from plotly import express  as px
import seaborn as sns


st.title ("Boston Housing")

#Load Boston Dataset
fname = "data//boston.csv"
df = pd.read_csv(fname)

st.image("data//boston_house.png")

#nav = st.sidebar.radio("Navigation",["Input", "Output"])
st.subheader("Predicting the price of a house in Boston")

    #Create Independent and Dependent Variables
X = df[['CRIM','CHAS','NOX', 'RM','AGE', 'DIS']]
y = df['MEDV']

    # Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=428)

    #Load Decision Trees
from sklearn.linear_model import LinearRegression
lin_regression = LinearRegression()

lrMod = lin_regression.fit(X, y)

v_room = st.number_input("Enter the number of bed rooms:",0.00, 10.00, step = 1.000, value= 3.00)
v_age = st.slider("Age of the property",0,100)
v_dist = st.slider("Distance from the office",0.0,15.0,step=0.5)
if st.checkbox("Next to Charls rever?"):
     v_ch = 1
else:
    v_ch = 0

v_crim = st.number_input("Enter the preferred crime rate:",0.00, 10.00, step = 0.100, value= 3.00)
v_NOX = st.number_input("Enter the NOX value in the neighborhood:",0.00, 1.00, step = 0.010, value= 0.10)

test_data = pd.DataFrame(
        dict(CRIM = v_crim,
            CHAS = v_ch,
            NOX = v_NOX,
            RM = v_room,
            AGE = v_age,
            DIS = v_dist),
        index=[0]
)
pred = lrMod.predict(test_data)
#add a streamlit button if pressed should navigate to output
if st.button("Predict"):
    
    st.success(f"Your predicted property price is $ {round (pred[0]*1000,2)}")
    st.write('---')

    st.write('Specified Input parameters')
    st.write(test_data)
    st.write('---')
    
    #st.write(test_data)


            
