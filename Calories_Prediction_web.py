# Code for predictiong the calories burnt and to be deploy over streamlit

import streamlit as st
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load the datasets
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Combine the datasets
Combined_Dataset = pd.concat([exercise, calories.iloc[:, 1]], axis=1)
Combined_Dataset.replace(["male", "female"], [0, 1], inplace=True)

# Prepare the data for training
x = Combined_Dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
y = Combined_Dataset.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize and train the model
cal = XGBRegressor()
cal.fit(x_train, y_train)

# Evaluate the model
train_score = cal.score(x_train, y_train) * 100
test_score = cal.score(x_test, y_test) * 100


st.title("Calories Prediction System")

st.write(f"Training Accuracy: {train_score:.2f}%")
st.write(f"Testing Accuracy: {test_score:.2f}%")

# Function for prediction
def CaloriesPrediction():
    Gender = st.selectbox("Enter Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    Age = st.number_input("Enter your Age", min_value=0, max_value=120)
    Height = st.number_input("Enter your Height (cm)", min_value=50.0, max_value=250.0)
    Weight = st.number_input("Enter your Weight (kg)", min_value=20.0, max_value=200.0)
    Duration = st.number_input("Enter your Exercise Duration (minutes)", min_value=0.0, max_value=300.0)
    Heart_Rate = st.number_input("Enter your Heart Rate", min_value=30.0, max_value=200.0)
    Body_Temp = st.number_input("Enter your Body Temperature (°C)", min_value=35.0, max_value=45.0)

    if st.button("Predict"):
        input_data = [[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]]
        res = cal.predict(input_data)
        st.success(f"Calories Predicted: {res[0]:.2f} Kcal")

# Call of a function
CaloriesPrediction()

st.write("Created by Mohd Amaan Khan")
