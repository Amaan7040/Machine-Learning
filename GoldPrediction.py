import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
gold_dataset = pd.read_csv("path/to/csvfile")
pd.set_option('display.max_rows', None) # to display all the rows
gold_dataset
gld = gold_dataset.GLD
gld.describe() #Description of GLD column
# Splitting the data into x and y
x = gold_dataset.iloc[:,[1,3,4,5]]
y = gold_data.iloc[:,2]
# Gives idea of Gold Price Density
sns.distplot(y,color="Blue")
# Spliiting Dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# Using RandomForestRegressor Algorithm
gold_model = RandomForestRegressor(n_estimators=7) # n_estimators=7 it can be changed according to the data
# Fitting the training dataset
gold_model.fit(x_train,y_train)
gold_model.predict(x_test)
#Training score
gold_model.score(x_train,y_train)*100
#Testing score
gold_model.score(x_test,y_test)*100
# Sample Prediction
gold_model.predict([[918.369995,38.849998,13.970000,1.394700]])
