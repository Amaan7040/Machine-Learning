# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
# Importing Dataset1
calories = pd.read_csv("calories.csv")
pd.set_option('display.max_rows', None)
calories
# Importing Dataset2
exercise = pd.read_csv("exercise.csv")
pd.set_option('display.max_rows', None)
exercise
# Combining both the dataset
Combined_Dataset = pd.concat([exercise,calories.iloc[:,1]],axis=1)
Combined_Dataset
# Getting information of a dataset
Combined_Dataset.info()
#Male => 0
#Female => 1
Combined_Dataset.replace(["male","female"],[0,1],inplace=True)
# Plotting graph of te dataset on the basis of Age Attribute
sns.distplot(Combined_Dataset['Age']) # Can also plot any other attribute graph as well.
# Statistical Measures of dataset
Combined_Dataset.describe()
# Splitting the dataset as dependent and independent data
x = Combined_Dataset.iloc[:,[1,2,3,4,5,6,7]]
y = Combined_Dataset.iloc[:,8]
# Splitting Dataset into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# Using XGBRegressor 
cal = XGBRegressor()
# Fitting the model
cal.fit(x_train,y_train)
# Predicting Testing Dataset
cal.predict(x_test)
#Training Accuracy
cal.score(x_train,y_train)*100
#Testing Accuracy
cal.score(x_test,y_test)*100
# Prediction Function
def CaloriesPrediction():
    name = input("Enter your name:")
    Gender = int(input("Enter Gender 0 for male and 1 for female:"))
    Age = float(input("Enter your Age :"))
    Height = float(input("Enter your Height :"))
    Weight = float(input("Enter your Weight :"))
    Duration = float(input("Enter your Duration :"))
    Heart_Rate = float(input("Enter your Heart_Rate :"))
    Body_Temp = float(input("Enter your Body_Temp :"))
    res = cal.predict([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
    print(f"{name} Calories Predicted is {res} Kcal.")
# Call of the function
CaloriesPrediction()
