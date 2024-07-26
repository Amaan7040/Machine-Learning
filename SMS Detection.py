import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk # Commonly used in NLP 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import seaborn as sns
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pickle
spam = pd.read_csv("SPAM.csv", encoding='latin1')
spam
# Data Cleaning
spam.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
spam.rename(columns={'v1':'Result','v2':'Content'},inplace=True)
encode = LabelEncoder()
spam['Result'] = encode.fit_transform(spam['Result'])
# O => Ham i.e Not Spam
# 1 => Spam
# Checking Null Values
spam.isnull().sum()
# Checkin duplicate values
spam.duplicated().sum()
spam = spam.drop_duplicates()
# EDA
plt.pie(spam['Result'].value_counts(),labels=['Ham(Not Spam)','Spam'],autopct="%.2f")
nltk.download('punkt')
# Adding 3 more columns for data analaysis 
spam["Numbers_of_characters"] = spam['Content'].apply(len)
spam["Numbers_of_words"] = spam['Content'].apply(lambda x:len(nltk.word_tokenize(x)))
spam["Numbers_of_sentences"] = spam['Content'].apply(lambda x:len(nltk.sent_tokenize(x))) 
# All the statistical measures
spam[["Numbers_of_characters","Numbers_of_words","Numbers_of_sentences"]].describe()
# Data Preprocessing or Transformation which consist of lowercase,tokenization,removing special characters,removing stopword & punctuation
# and Stemming
def Data_Tranformation(text):
#     lowercase
    text = text.lower()
#     tokenization
    text = nltk.word_tokenize(text)
#     removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
#     removing stopword & punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
#      Stemming
    p = PorterStemmer()
    text = y[:]
    y.clear()
    for i in text:
        y.append(p.stem(i))
    
    return " ".join(y)
# Model Building
tf = TfidfVectorizer(max_features=3400)
x = tf.fit_transform(spam["Transformed_Content"]).toarray()
y = spam.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
b = BernoulliNB()
sv = SVC(kernel="sigmoid",probability=True)
et = ExtraTreesClassifier(n_estimators=50)
vc = VotingClassifier(estimators=[('sv',sv),('bn',b),('etc',et)],voting='soft')
vc.fit(x_train,y_train)
# Training Acuracy
vc.score(x_train,y_train)*100
# Testing Accuracy
vc.score(x_test,y_test)*100
