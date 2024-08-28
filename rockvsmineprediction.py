#Importing Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection and Data Processing
#Loading the dataset to a pandas DataFrame
sonar_data = pd.read_csv('/content/SonarData.csv',header=None)

#Finding the Number of Rows and Columns in the Dataset
sonar_data.shape

#Statistical measures of the data
sonar_data.describe()

#Finding the number of Rocks and Mines in the dataset
sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

#Seperating data and labels
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]

#Training and Test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

#Model Training -> Logistic Regression
model = LogisticRegression()

#Training the model with training data
model.fit(X_train,Y_train)

#Model Evalutation -> Accuracy
#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Accuracy on training data : ",training_data_accuracy)

#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print("Accuracy on testing data : ",test_data_accuracy)

#MAKING A PREDICTIVE SYSTEM
#Give input to check for a particular instance
input_data = ()

#Changing the input_data to a numpy array
input_data_as_np_array  = np.asarray(input_data)

#Reshaping the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_np_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R' :
  print('The object is a Rock')
else :
  print('The object is a Mine')
