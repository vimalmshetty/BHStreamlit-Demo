import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import altair as alt #pip install altair
from plotly import graph_objs as go
from plotly import express  as px
import seaborn as sns

#streamlit hello

st.title ("Boston Housing")

# st.header("Header")

# st.subheader("Sub header")

# st.write("my text")

# st.markdown(""" 
# # h1 tag
# ## h2 tag
# ### h3 tag
# :moon:<br>
# :sunglasses: <br>
# **bold**
# _italics_
# """,True)

# st.latex(r''' a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#      \sum_{k=0}^{n-1} ar^k =
#      a \left(\frac{1-r^{n}}{1-r}\right)''')

# d = {
#     "name":"Harsh",
#     "language":"Python",
#     "topic":"Streamlit"
# } 

# st.write(d)

## Plots

#Load Boston Dataset
df = pd.read_csv("data//boston.csv")

st.image("data//boston_house.png")

nav = st.sidebar.radio("Navigation",["Home", "Prediction"])

if nav == 'Home':
    if st.checkbox("Show data"):
        #Show data
        st.dataframe(df)
        # st.table(df)
        # st.write(df)



    if st.checkbox("Show map"):
        val = st.slider("Filter data based on Median Value",0,40)
        fdata = df.loc[df["MEDV"]>= val]
        city_data = fdata[["LON","LAT","MEDV"]]
        city_data.columns = ['longitude','latitude', 'Medv']
        st.map(city_data)


    graph = st.selectbox("What kind of Graph?", ["Non-Interactive", "Interactive"])


    if graph == 'Non-Interactive':

        fig = sns.relplot(data = df,
                    x = 'LON',
                    y = 'LAT',
                    size = 'MEDV',
                    hue = 'MEDV',
                    kind = 'scatter',
                    s = 100,
                    aspect =3 )
        st.pyplot(fig)    

    if graph == 'Interactive':
        fig = go.Figure(px.scatter(df,
                        x = 'LON',
                        y = 'LAT',
                        size = 'MEDV').update_traces(mode="markers"))
        st.plotly_chart(fig)

if nav == 'Prediction':
    st.header("Prediction")

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

    #st.write(test_data)
    pred = lrMod.predict(test_data)

    if st.button("Predict"):
            st.success(f"Your predicted property price is $ {round (pred[0]*1000,2)}")
