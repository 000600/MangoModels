# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('mango_data.csv')
df = pd.DataFrame(df)

# Remove unnecessary columns from the dataset
df = df.drop(labels = ["No"], axis = 1) # These columns will not assist the model in learning how to predict the circumference of a mango

# Map non numeric values
df.Grade = df.Grade.map({'A': 0, 'B': 1})

# Scale x values
scaler = StandardScaler()
for col in df.columns:
  if col != 'Grade':
    df[col] = scaler.fit_transform(df[[col]])

# Initialize x and y lists
y = list(df.pop("Grade"))
x = df

# Divide the x and y values into two sets: train, and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

# Create and train model
model = SVC(C = 1.0, kernel = 'linear')
model.fit(x_train, y_train)

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
predictions = model.predict(x_test)
print(f"\nModel's Prediction on a Sample Input: {predictions[index]}")
print(f"Actual Label on the Same Input: {y_test[index]}")

# View test accuracy
test_acc = model.score(x_test, y_test)
print(f'\nTest accuracy: {test_acc * 100}%')

# Evaluate model
cr = classification_report(predictions, y_test)
print('\nClassification Report:')
print(cr)
