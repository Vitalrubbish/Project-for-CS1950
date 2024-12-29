import matplotlib.pyplot as plt
import random

n = 0
sample_x = []
sample_y = []
A, B, C, D, E, F = 0, 0, 0, 0, 0, 0
avg_x, avg_y = 0, 0

def read_sample():
    global n, sample_x, sample_y, avg_x, avg_y
    print("Please input the number of samples. (No more than 100)")
    n = int(input())
    for i in range(1, n + 1):
        print(f"Please input x and y of the sample {i}")
        x, y = map(float, input().split())
        sample_x.append(x)
        sample_y.append(y)
        avg_x += x
        avg_y += y
    avg_x /= n
    avg_y /= n

def error_function():
    global A, B, C, D, E, F
    A, B, C, D, E, F = 0, 0, 0, 0, 0, 0
    for i in range(n):
        x, y = sample_x[i], sample_y[i]
        A += x * x
        B += 1
        C += 2 * x
        D -= 2 * x * y
        E -= 2 * y
        F += y * y

def gradient_descent():
    global A, B, C, D, E, F
    print("Please input the learning_rate: ")
    alpha = float(input())
    print("Please input the initial a and b: ")
    a_, b_ = map(float, input().split())
    print("Please input the number of iterations: ")
    iteration_cnt = int(input())

    losses_1 = []
    losses_2 = []
    interation = []


    a = a_
    b = b_
    for i in range(iteration_cnt):

        loss = 0
        for j in range(n):
            x, y = sample_x[j], sample_y[j]
            y_pred = a * x + b
            loss += (y_pred - y) ** 2
        losses_1.append(loss)
        interation.append(i + 1)

        partial_a = 2 * A * a + C * b + D
        partial_b = 2 * B * b + C * a + E

        a = a - alpha * partial_a / n
        b = b - alpha * partial_b / n

    print("The a and b calculated by BGD")
    print(f"a = {a:.4f} b = {b:.4f}")

    a = a_
    b = b_
    for i in range(iteration_cnt):
        rand = random.randint(0, n - 1)
        loss = 0
        for j in range(n):
            x,y = sample_x[j], sample_y[j]
            y_pred = a * x + b
            loss += (y_pred - y) ** 2
        losses_2.append(loss)

        x,y = sample_x[rand], sample_y[rand]
        partial_a = 2 * (y - a * x - b) * (-x)
        partial_b = -2 * (y - a * x - b)
        a = a - alpha * partial_a
        b = b - alpha * partial_b

    print("The a and b calculated by SGD")
    print(f"a = {a:.4f} b = {b:.4f}")

    x = interation
    plt.subplot(2, 1, 1)
    plt.plot(x, losses_1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('BGD')

    plt.subplot(2, 1, 2)
    plt.plot(x, losses_2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('SGD')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.show()

def calculate():
    global A, B, C, D, E, F, avg_x, avg_y
    a_0 = (-D / 2 - n * avg_x * avg_y) / (A - n * avg_x * avg_x)
    b_0 = avg_y - a_0 * avg_x
    print("The a and b calculated by least square method:")
    print(f"a = {a_0:.4f} b = {b_0:.4f}")


read_sample()
error_function()
gradient_descent()
calculate()