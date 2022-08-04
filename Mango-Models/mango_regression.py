# Imports
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Data.csv')
df = pd.DataFrame(df)

# Remove unnecessary columns from the dataset
df = df.drop(labels = ["No", "Grade", "Length"], axis = 1) # These columns will not assist the model in learning how to predict the circumference of a mango

# Scale x values
scaler = StandardScaler()
for col in df.columns:
  if col != 'Circumference':
    df[col] = scaler.fit_transform(df[[col]])

# Initialize x and y lists
x = []
y = list(df.pop("Circumference"))
    
# Add dataset to x and y lists
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0])): # Loop through all columns
    rows.append(df.iloc[row][point])
  x.append(rows)

# Divide the x and y values into three sets: train, test, and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1)

# Create and train model
model = XGBRegressor(n_estimators = 7000, learning_rate = 0.001)
model.fit(x_train, y_train, early_stopping_rounds = 5, eval_set = [(x_val, y_val)], verbose = 1) # Predicts the circumference of a mango

# View mean squared error of the model
predictions = model.predict(x_test)
mse = mean_squared_error(predictions, y_test)
print("\nMean Squared Error on Test Dataset (MSE):", mse)

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict([x_test[index]])[0]
print(f"Model's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}")

# Calculate model's approximate deviation
error = []
for val in range(len(x_test)): # Loop through test values and have model predict on those test values
  error_val = abs(model.predict([x_test[val]]) - y_test[val])[0] # Determine the difference between the model's predicted labels and actual labels
  error.append(float(error_val)) # Store difference values in a list for plotting

# Visualize positive difference
y_pos = np.arange(len(error))

plt.figure(figsize = (8, 6))
plt.bar(y_pos, error, align = 'center')
plt.ylabel('Positive Difference Between Predicted and Actual Values')
plt.xlabel('Input Index')
plt.title('XGBoost Regression Error')

plt.show()

# View model's train and test predictions compared to actual values
predictions_train = model.predict(x_train)
predictions_test = model.predict(x_test)

# Create x-values necessary in scatterplot
x_train_plot = [i for i in range(len(y_train))]
x_test_plot = [i for i in range(len(y_test))]

# Plot train predictions compared to train values
plt.scatter(x_train_plot, y_train, label = 'Actual Mango Circumference')
plt.scatter(x_train_plot, predictions_train, label = 'Predicted Mango Circumference')
plt.xlabel('Data Entry Number')
plt.ylabel('Circumference (cm)')
plt.title("Model's Predicted Circumference Compared to Actual Circumference (Training Data)")
plt.legend()
plt.show()

# Plot train predictions compared to train values
plt.scatter(x_test_plot, y_test, label = 'Actual Mango Circumference')
plt.scatter(x_test_plot, predictions_test, label = 'Predicted Mango Circumference')
plt.xlabel('Data Entry Number')
plt.ylabel('Circumference (cm)')
plt.title("Model's Predicted Circumference Compared to Actual Circumference on (Testing Data)")
plt.legend()
plt.show()