# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Load the sonar dataset
sonar = pd.read_csv("sonardata.csv")

# Set pandas options to display all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Display the dataset
print(sonar)

# Get information about the dataset
print(sonar.info())

# Get statistical summary of the dataset
print(sonar.describe())

# Split the dataset into features (X) and target (y)
# Features are all columns except the last one
x = sonar.iloc[:, :-1]
# Target is the last column
y = sonar.iloc[:, 60]

# Split the data into training and testing sets
# Use stratify to maintain the distribution of target classes in both sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Initialize the Extra Trees Classifier with 50 estimators
etc = ExtraTreesClassifier(n_estimators=50, random_state=42)

# Fit the model on the training data
etc.fit(x_train, y_train)

# Predict on the test data
y_pred = etc.predict(x_test)

# Calculate and print the training accuracy
train_accuracy = etc.score(x_train, y_train) * 100
print(f'Training Accuracy: {train_accuracy:.2f}%')

# Calculate and print the testing accuracy
test_accuracy = etc.score(x_test, y_test) * 100
print(f'Testing Accuracy: {test_accuracy:.2f}%')
