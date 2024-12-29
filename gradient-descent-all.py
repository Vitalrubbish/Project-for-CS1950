import numpy as np
import matplotlib.pyplot as plt

n = int(input("Please input the number of points: "))
x = []
y = []
print("Please input points: ")
for i in range(n):
    xi, yi = map(float, input(f"Point {i + 1} like \"x y\": ").split())
    x.append(xi)
    y.append(yi)
x, y = np.array(x), np.array(y)

def calc_gradients(w0, w1):
    predictions = w0 * x + w1
    grad_w0 = -2 * np.sum(x * (y - predictions)) / n
    grad_w1 = -2 * np.sum(y - predictions) / n
    return grad_w0, grad_w1

print("Please input learning rate:")
learning_rate = float(input())
print("Please input iterations:")
iterations = int(input())

print("Please input starting line: y = ax + b: \"a b\":")
w0, w1 = map(float, input().split())

for t in range(iterations):
    grad_w0, grad_w1 = calc_gradients(w0, w1)
    w0 -= learning_rate * grad_w0
    w1 -= learning_rate * grad_w1
    # print(w0, w1)
    # print(grad_w0, grad_w1)

print(f"The linear regression line: y = {w0:.2f}x + {w1:.2f}")
print(f"Gradient: {calc_gradients(w0, w1)}")

plt.scatter(x, y, color = 'blue', label = 'Data points')
delta = (max(x) - min(x)) / 20
x_line = np.linspace(min(x) - delta, max(x) + delta, 400)
y_line = w0 * x_line + w1
plt.plot(x_line, y_line, color = 'red', label = f'Fitted line: y = {w0:.2f}x + {w1:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()