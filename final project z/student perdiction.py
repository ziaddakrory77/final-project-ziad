import numpy as np
import pandas as pd #reading and working with data (like csv)
import matplotlib.pyplot as plt #for plotting graph and visualizing data
from sklearn.model_selection import train_test_split #splitting data into training and testing sets
from sklearn.linear_model import LinearRegression #the machine learning model
from sklearn.metrics import mean_squared_error, r2_score #metrics to evaluate the model
data= pd.read_csv('student_data.csv', sep=';') # Load the dataset with semicolon as separator
data = data.dropna()  # Remove rows with missing values
X = data[['Study_Hours']]  # Features
Y = data['Grade']  # Target variable
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Splitting the data into training and testing sets
#train model
model = LinearRegression()
model.fit(x_train, y_train)
#predect student grades
y_pred = model.predict(x_test)
#evaluation
print("MSE:" , mean_squared_error (y_test, y_pred))
print("R2 Score:", r2_score(y_test,y_pred))
#visualization
plt.scatter(y_test, y_pred, color='red', )
plt.xlabel('Grades')
plt.ylabel('Study_hours')
plt.figure(figsize = (10,6))
plt.title('Student grade prediction')
plt.grid(True)
data_sorted = data.sort_values(by='Study_Hours') 
plt.plot(data_sorted['Study_Hours'], data_sorted['Grade'], color='red', linestyle='-', marker='x')
plt.show()
#user input
hours= float(input("Enter study hours:"))
predicted_grade = model.predict([[hours]])
print(f"predicted grade for { hours} hours of study:{predicted_grade[0]:.2f}")
