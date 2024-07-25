# Importing required liberaries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# Importing Dataset
diabetes_data = pd.read_csv("diabetes.csv")
pd.set_option("display.max_rows",None)
diabetes_data
# 1 => diabetic
# 0 => non-diabetic
# Information about dataset
diabetes_data.info()
# Statistical Measures
diabetes_data.describe()
# mean value of diabetic and non diabetic person
diabetes_data.groupby("Outcome").mean()
#  Splitting Data as a dependent and independent data 
x = diabetes_data.iloc[:,:-1]
y = diabetes_data.iloc[:,-1]
# standardizing the dataset
std = StandardScaler()
std_data = std.fit_transform(x)
std_data
X = std_data
# Splitting the data as train and test data
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# Using RandimForesClassifier Algorithm
rn = RandomForestClassifier(n_estimators=250)
# Fitting the model
rn.fit(x_train,y_train)
# Predicting the model
rn.predict(x_test)
# Training Accuracy
rn.score(x_train,y_train)*100
# Testing Accuracy
rn.score(x_test,y_test)*100
# Prediction function or model
def DiabetesPrediction():
    Name = input("Enter your name:")
    Age = int(input("Enter your age (in integers):"))
    Pregnancies = int(input("How many Times do you get pregnant?:"))
    Glucose = int(input("Enter your Glucose Level:"))
    BloodPressure = int(input("Enter your BloodPressure Level:"))
    SkinThickness = int(input("Enter your skin thickness:"))
    Insulin = int(input("Enter your Insulin Level:"))
    BMI =  float(input("Enter your Body Mass Index:"))
    DiabetesPedigreeFunction = float(input("Enter your family history score in diabetes(DiabetesPedigreeFunction):"))
    input_data = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
    stdD = std.fit_transform(input_data)
    res = rn.predict(stdD)
    if res[0] == 0:
        print(f"{Name} you do not have diabetes.")
    elif res[0] == 1:
        print(f"{Name} you have diabetes.Go to your nearest hospital for treatment.")
    else:
        print("Unable to predict or detect the right data for the moment.Please Try Again.")
# Call of a Prediction Function or Model
DiabetesPrediction()
