#3*x^2+2x+4sin(x)
#6x+2+4cos(x)

import math
import numpy as np
import matplotlib.pyplot as plt

# Hàm tính gradient
def grad(x):
    return 6*x + 2 + 4*np.cos(x)

# Hàm tính giá trị hàm chi phí
def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)

# Thuật toán Gradient Descent
def myGD1(eta, x0):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - eta * grad(x[-1])  # Cập nhật giá trị x mới
        if abs(grad(x_new)) < 1e-3:  # Dừng khi gradient rất nhỏ
            break
        x.append(x_new)  # Thêm giá trị mới vào danh sách
    return (x, it)

# Tham số và khởi tạo
eta = 0.01  # Tốc độ học (learning rate)
x0 = -1  # Giá trị khởi tạo

# Chạy Gradient Descent
(x_values, num_iterations) = myGD1(eta, x0)

# In kết quả
print("Giá trị cuối cùng của x:", x_values[-1])
print("Số lần lặp:", num_iterations)
p = cost(x_values[-1])
print("cost:", p)
# Vẽ biểu đồ hàm chi phí
x_range = np.linspace(-10, 10, 100)
cost_values = cost(x_range)

plt.plot(x_range, cost_values, label='Cost function')
plt.scatter(x_values, [cost(x) for x in x_values], color='red', label='Gradient Descent steps')
plt.title('Gradient Descent on Cost Function')
plt.xlabel('x')
plt.ylabel('Cost')
plt.legend()
plt.show()

