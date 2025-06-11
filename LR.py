import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('student_scores.csv')
X = df[['hours']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("(MSE):", mse)
print("(R² Score):", r2)

prediction = model.predict(pd.DataFrame([[7]], columns=['hours']))

print("y ", prediction[0])

plt.scatter(X, y, color='blue', label='Original Data')          # Plot the original data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot the regression line
plt.xlabel("Study Hours")                                       # X-axis label
plt.ylabel("Student Score")                                     # Y-axis label
plt.title("Linear Regression")                                  # Title of the graph
plt.legend()                                                    # Show legend
plt.grid(True)                                                  # Show grid
plt.show()                                                      # Display the plot



