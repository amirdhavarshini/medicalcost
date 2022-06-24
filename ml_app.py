import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
@st.cache
def get_data():
        data = pd.read_csv(r"C:\Users\Amirdha Selvaraj\Downloads\insurance.csv")
        return data


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
    st.title("Heyy, hello and Welcome to my End to End ML Model on Medical Insurance Cost")
    st.write('''This project is based on Prediction by using Machine Learning Algorithms.
    In this project i have predicted the Medical Insurance cost.''')
    st.subheader('Introduction')
    st.write('''The goal of this project is to allows a person to get an idea about the necessary amount required according to their own health status. 
    Later they can comply with any health insurance company and their schemes & benefits keeping in mind the predicted amount from our project. 
    This can help a person in focusing more on the health aspect of an insurance rather than the futile part.\n Our project does not give the exact amount required for any health insurance company but gives enough idea about the amount associated with an individual for his/her own health insurance.''')

with dataset:
    st.subheader("* **Medical Cost Prediction** *")
    st.write("I took this dataset from Kaggle.")
    st.write('''The data was in structured format and was stores in a csv file.  Dataset is not suited for the regression to take place directly. So cleaning of dataset becomes important for using the data under various regression algorithms.
    In a dataset not every attribute has an impact on the prediction. Whereas some attributes even decline the accuracy, so it becomes necessary to remove these attributes from the features of the code. 
    Removing such attributes not only help in improving accuracy but also the overall performance and speed.''')
    st.write('''After preparing the dataset, models would be instantiated for the model fitting and prediction part.Random seed are applied throughtout the model fitting process for reproducibility. Regressors are used in this experiment as regression is more suitable for the objective of the task (prediction of medical insurance charges).''')
    data = get_data()
    st.write(data.head(10))

with features:
        st.header("Features")
        st.write("The dataset is comprised of 1338 records with 6 attributes. Attributes are as follow age, gender, bmi, children, smoker and charges")
        st.subheader('Charges in medical Insurance')
       
        charges = pd.DataFrame(data["charges"].value_counts()).head(50)
        st.line_chart(data["charges"])

with modelTraining:
        st.header("Now lets predict the Medical Insurance Cost")
        sel_col, disp_col= st.columns(2)
        st.write("Here you can choose the parameters and can check your Medical Insurance cost")
        sel_col, disp_col = st.columns(2)
        age  = sel_col.slider('What is your Age?', min_value = 1, max_value = 100, value = 10, step = 5)
        age = int(age)
        bmi =sel_col.slider('What is your BMI?', min_value = 15, max_value = 100, value = 5, step = 5)
        bmi = float(bmi)
        sel_col.text('Here is a list of features in my data')
        sel_col.write(data.columns)
        charges = sel_col.text_input('What are the charges? : ',0)
        charges = float(charges)
        y = [['age']]
        X = [['charges']]
        
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X,y)
        pred = model.predict()
        mae(y, pred)
        new_data = [['age','bmi','charges']]
        predict_value = model.predict(new_data)
result = st.button("Predict")
if predict_value == 1:
            st.subheader('The predicted medical cost is: ')
else:
     st.subheader('Could not able to predict medical cost: ')
