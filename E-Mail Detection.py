import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# importing the dataset
mail = pd.read_csv("path/to/dataset")
pd.set_option('display.max_rows', None)
mail
#Checking is there any null value present in a dataset ot not.
mail.isnull().sum()
#Replace spam with 0 and Ham with 1
#Spam i.e 0 is a fraud mail and ham i.e 1 is a normal mail.
mail.replace(["spam","ham"],[0,1],inplace=True)
# Seperation of data 
x = mail.iloc[:,1]
y = mail.iloc[:,0]
# Getting Statistical measure of Cateory Column
m = mail.Category
m.describe()
# Spliiting data into train and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# Extraction features from the text in order to fit the model better and efficiently basically convert all he textual content into numerical content for easy undersanding of the machinr
extraction = TfidfVectorizer(max_df=1,stop_words="english",lowercase=True)
x_train_fetaure_extraction = extraction.fit_transform(x_train)
x_test_fetaure_extraction  = extraction.transform(x_test)
# Using Algorithm
from sklearn.ensemble import RandomForestClassifier
mail_detect = RandomForestClassifier(n_estimators=30)
# Fitting the training data
mail_detect.fit(x_train_fetaure_extraction,y_train)
# Prdicting testing data
mail_detect.predict(x_test_fetaure_extraction)
# Training Accuracy
mail_detect.score(x_train_fetaure_extraction,y_train)*100
# Testing Accuracy
mail_detect.score(x_test_fetaure_extraction,y_test)*100
# Mail Prediction Function or setup
def MailPrediction(content=[""]):
    inputMail = content
    input_feature = extraction.transform(inputMail)
    pred = mail_detect.predict(input_feature)
    print(pred)
    if pred[0] == 1:
        print("Email is not a spam mail.")
    elif pred[0] == 0:
        print("Email is a spam mail.")
    else:
        print("Not able to detect.Sorry")
# Call of the function
MailPrediction(content=[""]) # put the mail content in between the " ".
