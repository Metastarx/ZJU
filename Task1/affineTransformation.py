import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 8)

# 定义三角形的顶点
X = np.array([[0.0, 1.0, 0.5, 0.0],
              [0.0, 0.0, np.sqrt(3)/2, 0.0]])

# 初始旋转角度和旋转矩阵
theta_step = np.pi / 4  # 每步旋转45度
theta = 0.0  # 初始角度

for t in range(4):  # t = 0, 1, 2, 3
    # 当前时间步的旋转矩阵
    Rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    # 应用旋转
    Y = Rotation.dot(X)

    # 绘图
    plt.plot(Y[0, :], Y[1, :], label=f't = {t}')

    # 更新角度为下一个时间步
    theta += theta_step

plt.title('Triangle with Rotation over Time')
plt.axis('equal')
plt.legend()
plt.show()
