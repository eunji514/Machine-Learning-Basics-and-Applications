import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gradient Descent
def gradientDescent(x, y, learning_rate, num_iterations):
    m = len(x)
    a, b = 0, 0

    for i in range(num_iterations):
        y_pred = a * x + b
        gradient_a = (1/m) * np.sum(x * (y_pred - y))
        gradient_b = (1/m) * np.sum(y_pred - y)

        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b

        if (i + 1) % 100 == 0:
            cost = (1/(2*m)) * np.sum((a * x + b - y)**2)
            print(f'Iteration {i+1}, Cost: {cost:.4f}, a {a}, b {b}')
    
    return a, b

# Normal Equation
def normalEquation(x, y):
    X = np.column_stack((np.ones_like(x), x, x**2))
    params = np.linalg.inv(X.T @ X) @ X.T @ y
    return params[2], params[1], params[0]

#  Plot of the linear regression model
def plot_linear_regression(x, y, a, b):
    plt.scatter(x, y, label='Data')
    plt.plot(x, a * x + b, color='orange', label=f'Linear Regression: y = {a:.4f}x + {b:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

#  Plot of the quadratic regression model with a single line
def plot_quadratic_regression(x, y, a, b, c):
    plt.scatter(x, y, label='Data')
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(x_range, a * x_range**2 + b * x_range + c, color='purple', label=f'Quadratic Regression: y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Load the dataset
data = pd.read_csv('data_hw1.csv')
x = data['x']
y = data['y']

# Task 1: Linear Regression with Gradient Descent
learning_rate = 0.01
num_iterations = 1000
a, b = gradientDescent(x, y, learning_rate, num_iterations)

# Print and plot results for Task 1
print(f'Task 1 - Linear Regression Parameters: a = {a:.4f}, b = {b:.4f}')
plot_linear_regression(x, y, a, b)

# Task 2: Quadratic Regression with Normal Equation
a, b, c = normalEquation(x, y)

# Print and plot results for Task 2
print(f'Task 2 - Quadratic Regression Parameters: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}')
plot_quadratic_regression(x, y, a, b, c)
