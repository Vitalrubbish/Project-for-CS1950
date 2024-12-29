import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

print("Please input f(x) = ")
input_func = input()
if '=' in input_func:
    input_func = input_func.split('=')[1].strip()

x = sp.symbols('x')
f_expr = sp.parse_expr(input_func)
df_expr = sp.diff(f_expr, x)

f = sp.lambdify(x, f_expr, 'numpy')
df = sp.lambdify(x, df_expr, 'numpy')

print("Please input start point:")
start_x = float(input())
print("Please input learning rate:")
learning_rate = float(input())
print("Please input iterations:")
iterations = int(input())
print("Please input considering ranges like \"a b\":")
ranges = list(map(float, input().split()))

# gradient-descent
x_vals = [] 
end_x = start_x
for t in range(iterations):
    if len(x_vals) :
        gradient = df(x_vals[-1])
        new_x = x_vals[-1] - learning_rate * gradient
    else :
        gradient = df(start_x)
        new_x = start_x - learning_rate * gradient
    # if new_x > ranges[1] : new_x = ranges[1]
    # if new_x < ranges[0] : new_x = ranges[0]
    if t != iterations - 1 : x_vals.append(new_x)
    else : end_x = new_x
    # print(new_x, f(new_x))
    # print(df(new_x))

#output answer
print(f"Minimum Point: ({x_vals[-1]:.2f}, {f(x_vals[-1]):.2f})")
print(f"Delta: {df(x_vals[-1])}")

#painting
x = np.linspace(ranges[0], ranges[1], 400) # 生成等间距 x 值用于画图
y = f(x) # 根据 x 值生成 y
plt.plot(x, y, label=r'$y = f(x)$', color = 'blue') # 函数曲线，蓝色

plt.scatter(x_vals, f(np.array(x_vals)), color = 'red', label = 'Gradient Descent Path') # 过程散点，红色
plt.scatter(start_x, f(start_x), color = 'green', label = 'Start Point') # 起始点
plt.scatter(end_x, f(end_x), color = 'orange', label = 'End Point') # 终止点
plt.plot(x_vals, f(np.array(x_vals)), color = 'red', linestyle = '--', alpha = 0.5) # 散点折线，虚线
plt.plot([start_x, x_vals[0]], [f(start_x), f(x_vals[0])], color = 'red', linestyle = '--', alpha = 0.5)
plt.plot([end_x, x_vals[-1]], [f(end_x), f(x_vals[-1])], color = 'red', linestyle = '--', alpha = 0.5)

plt.title('Gradient Descent on $y = f(x)$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend() # 图例
plt.grid(True) # 开启网格
plt.show()
