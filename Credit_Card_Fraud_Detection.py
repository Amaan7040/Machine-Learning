import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("path/to/your_file",)
data
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
#Description of legal transactions
legit.Amount.describe()
#Description of fraud transactions
fraud.Amount.describe()
x = data.iloc[:,:-1]
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
detect = LogisticRegression()
detect.fit(x_train,y_train)
detect.predict(x_test)
detect.score(x_train,y_train)*100
detect.score(x_test,y_test)*100
