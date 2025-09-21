import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#title
st.title("web app using streamlit")

#Image 
st.image("streamlit_img.png",width=200)

st.title("case study on diamond dataset")

data = sns.load_dataset("diamonds")
st.write("Shape of a dataset",data.shape)
menu = st.sidebar.radio("menu",["Home","prediction price"])
if menu =="Home":
    st.image("diamond.webp")
    st.header("Tabular data of a diamond")
    if st.checkbox("Tabular Data"):
        st.table(data.head(50))
    st.header("Statistical summary of a Dataframe")
    if st.checkbox("Statistics"):
        st.table(data.describe())
    if st.header("Correlation graph"):
        fig,ax = plt.subplots(figsize=(5,2.5))
        sns.heatmap(data.corr(),annot = True,cmap="coolwarm")
        st.pyplot(fig)

    st.title ("Graphs")
    graph = st.selectbox("Different types of graphs",["Scatter plot","Bar graph","Histogram"])
    if graph == "Scatter plot":
        value = st.slider("Filter data using carat",0,6)
        data = data.loc[data["carat"]>=value]
        fig,ax=plt.subplots(figsize=(10,5))
        sns.scatterplot(data=data,x="carat",y="price",hue ="cut")
        st.pyplot(fig)
    if graph == "Bar graph":
        fig,ax=plt.subplots(figsize=(3.5,2))
        sns.barplot(x="cut",y=data.cut.index,data=data)
        st.pyplot(fig)
    if graph == "Histogram":
        fig,ax=plt.subplots(figsize=(5,3))
        sns.distplot(data.price,kde=True)
        st.pyplot(fig)    

if menu == "prediction price":
    st.title("Prediction Price of a Diamond")
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    x = np.array(data["carat"]).reshape(-1,1)
    y = np.array(data["price"]).reshape(-1,1)
    lr.fit(x,y)
    value = st.number_input("carat",0.20,5.01,step=0.15)
    value = np.array(value).reshape(1,-1)
    prediction = lr.predict(value)[0]
    if st.button("Price prediction($)"):
        st.write(f"{prediction}")