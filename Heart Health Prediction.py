import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
Heart_data = pd.read_csv("heart_disease.csv")
pd.set_option("display.max_rows",None)
Heart_data
# Male => 1
# Female => 0
# Statistical measure of data
Heart_data.describe()
# Get the number of 0 i.e Healty Heart & 1 i.e Unhealthy Heart present in target attribute of a dataset
Heart_data["target"].value_counts()
# Splitting dataset into dependent and independent variable
x = Heart_data.iloc[:,:-1]
y = Heart_data.iloc[:,13]
# Training & Testing Data Split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
heart = RandomForestClassifier(n_estimators=100)
heart.fit(x_train,y_train)
# Predicting the testing data
heart.predict(x_test)
# Training Accuracy
heart.score(x_train,y_train)*100
# Testing Accuracy
heart.score(x_test,y_test)*100
# Prediction Function
def HeartHealthPrediction():
    name = input("Enter yoyr full name:")
    age = float(input("Enter your age:"))
    sex = int(input("Enter your sex (1 for male & o for female):"))
    cp = int(input("Enter your chest pain type (0-2):"))
    trestbps = float(input("Enter your resting blood pressure (in mm Hg on admission to the hospital):"))
    chol = float(input("Enter your serum cholestoral in mg/dl:"))
    fbs = int(input("Enter your fasting blood sugar &gt; 120 mg/dl(1 = true; 0 = false):"))
    restecg = int(input("Enter your resting electrocardiographic results (values 0,1,2):"))
    thalach = float(input("Enter your maximum heart rate achieved:"))
    exang = int(input("Enter your exercise induced angina (1 = yes; 0 = no):"))
    oldpeak = float(input("Enter your ST depression induced by exercise relative to rest:"))
    slope = float(input("Enter your the slope of the peak exercise ST segmen:"))
    ca = int(input("Enter your number of major vessels (0-3) colored by flourosopy:"))
    thal = int(input("Enter your thal: 0 = normal; 1 = fixed defect; 2 = reversable defect:"))
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    res = heart.predict(input_data)
    if res[0] == 0:
        print(f"{name} you have a healthy heart.Thus you did not have any Heart Disease.")
    elif res[0] == 1:
        print(f"{name} you have an unhealthy heart.Thus you have a Heart Disease and need immediate treatment.")
    else:
        print("Unable to predict due to some unexpected error.")
# Call odf a prediction function
HeartHealthPrediction()
