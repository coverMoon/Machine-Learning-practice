import numpy as np
import matplotlib.pyplot as plt

# 设置 Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
# [密度，含糖率，标签]
watermelon_data = np.array([
    [0.697, 0.460, 1],
    [0.774, 0.376, 1],
    [0.634, 0.264, 1],
    [0.608, 0.318, 1],
    [0.556, 0.215, 1],
    [0.403, 0.237, 1],
    [0.481, 0.149, 1],
    [0.437, 0.211, 1],
    [0.666, 0.091, 0],
    [0.243, 0.267, 0],
    [0.245, 0.057, 0],
    [0.343, 0.099, 0],
    [0.639, 0.161, 0],
    [0.657, 0.198, 0],
    [0.360, 0.370, 0],
    [0.593, 0.042, 0],
    [0.719, 0.103, 0]
])

# 分离特征和标签
X = watermelon_data[:, :2]
y = watermelon_data[:, 2]

# 按类别分离数据
# 类别0
X0 = X[y == 0]
# 类别1
X1 = X[y == 1]

# 计算均值向量
mu0 = np.mean(X0, axis=0)
mu1 = np.mean(X1, axis=0)

# 计算类内散度矩阵

# 计算坏瓜类别（X0）的散度矩阵 S0
S0 = (len(X0) - 1) * np.cov(X0.T)
# 计算好瓜类别（X1）的散度矩阵 S1
S1 = (len(X1) - 1) * np.cov(X1.T)

Sw = S0 + S1

# 计算类间散度矩阵
diff_means = mu1 - mu0
Sb = np.outer(diff_means, diff_means)

# 求解最佳投影方向
inv_Sw = np.linalg.inv(Sw)

target_matrix = inv_Sw.dot(Sb)

eigenvalues, eigenvectors = np.linalg.eig(target_matrix)

print("计算得到的特征值:\n", eigenvalues)
print("\n计算得到的特征向量矩阵:\n", eigenvectors)

# 找到最大特征值对应的特征向量
w = eigenvectors[:, np.argmax(eigenvalues)]

print("\n最大特征值对应的特征向量 (即最佳投影方向 w):\n", w)

# 结果可视化

# 创建一个新的图形
plt.figure(figsize=(10, 7))
# 绘制原始数据点
# 类别0 (坏瓜), 用蓝色圆点表示
plt.scatter(X0[:, 0], X0[:, 1], c='b', marker='o', label='坏瓜')
# 类别1 (好瓜), 用红色三角表示
plt.scatter(X1[:, 0], X1[:, 1], c='r', marker='^', label='好瓜')

# 为了让投影直线穿过数据中心，计算所有数据的均值点
data_mean = np.mean(X, axis=0)

# 创建两个点来定义这条直线，以便绘图
line_x = np.linspace(min(X[:, 0]) - 0.1, max(X[:, 0]) + 0.1, 200)

# 根据 w 计算直线的 y 值
# 直线的斜率是 w[1] / w[0]
# 使用点斜式方程: y - y_mean = slope * (x - x_mean)
slope = w[1] / w[0]
line_y = slope * (line_x - data_mean[0]) + data_mean[1]

# 绘制投影直线
plt.plot(line_x, line_y, 'g-', lw=2, label='linear_discriminant_analysis 投影方向')

# 添加图表标题和标签
plt.title('线性判别分析 (linear_discriminant_analysis) 在西瓜数据集上的最终结果', fontsize=16)
plt.xlabel('密度', fontsize=12)
plt.ylabel('含糖率', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.show()