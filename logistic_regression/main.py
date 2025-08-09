import numpy as np
import matplotlib.pyplot as plt

# 指定默认字体为“黑体”
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 手动录入数据
# 格式：[密度，含糖率，好瓜(1)或坏瓜(0)]
watermelon_data = [
    [0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1],
    [0.556, 0.215, 1], [0.403, 0.237, 1], [0.481, 0.149, 1], [0.437, 0.211, 1],
    [0.666, 0.091, 0], [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
    [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0], [0.593, 0.042, 0],
    [0.719, 0.103, 0]
]

# 将数据转换为 NumPy 数组
dataset = np.array(watermelon_data)

# 将数据集切分为特征 X 和标签 y
X = dataset[:, :2]
y = dataset[:, 2]


# 定义 Sigmoid 函数
def sigmoid(z):
    """
    计算 Sigmoid 函数的值
    :param z: 一个数值或一个 NumPy 数组
    :return:  sigmoid(z) 的值，与 z 的维度相同
    """
    return 1.0 / (1.0 + np.exp(-z))


# 初始化参数

# 获取特征数量
num_features = X.shape[1]

# 初始化权重 w 为一个 (num_features, 1) 的零向量
w = np.zeros((num_features, 1))

# 初始化偏置 b 为 0
b = 0

# 将 y 从 (17,) 变形为 (17, 1)
y = y.reshape(-1, 1)

# 训练模型

# 定义超参数
learning_rate = 0.9
num_iterations = 1000

# 获取样本数量 m
m = X.shape[0]

# --- 开始训练循环 ---
for i in range(num_iterations):
    # 正向传播：计算预测值
    # z = X·w + b
    z = np.dot(X, w) + b
    # y_hat = sigmoid(z)
    y_hat = sigmoid(z)

    # 计算损失
    loss = -1 / m * np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))

    # 反向传播：计算梯度
    dw = (1 / m) * np.dot(X.T, y_hat - y)
    db = (1 / m) * np.sum(y_hat - y)

    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 每隔100次迭代就打印一次损失值，看看它是不是在下降
    if (i + 1) % 100 == 0:
        print(f"迭代次数 {i + 1}/{num_iterations}，损失值 Loss: {loss:.4f}")

# 结果分析与可视化

# 使用训练好的参数进行预测
# 计算每个样本是“好瓜”的概率
probabilities = sigmoid(np.dot(X, w) + b)
# 根据概率进行分类，阈值设为 0.5
# 如果概率 >= 0.5，则预测为 1 (好瓜)，否则为 0 (坏瓜)
predictions = (probabilities >= 0.5).astype(int)

# 计算并打印准确率
# np.mean(predictions == y) 会计算预测正确样本的比例
accuracy = np.mean(predictions == y) * 100
print(f"\n模型在训练集上的准确率: {accuracy:.2f}%")

# 可视化结果
print("正在生成结果图像...")

# 创建一个图形
plt.figure(figsize=(10, 6))

# 绘制所有的数据点
# 先画出所有好瓜 (y=1) 的点，用 'o' 表示
plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='green', marker='o', label='好瓜 (真实)')
# 再画出所有坏瓜 (y=0) 的点，用 'x' 表示
plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='red', marker='x', label='坏瓜 (真实)')

# 绘制决策边界
x1_boundary = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
x2_boundary = -(w[0] * x1_boundary + b) / w[1]

plt.plot(x1_boundary, x2_boundary, 'b-', label='决策边界')

# 添加图例、标题和坐标轴标签
plt.title('西瓜数据集 3.0α - 对率回归决策边界')
plt.xlabel('密度 (Density)')
plt.ylabel('含糖率 (Sugar Content)')
plt.legend()
plt.grid(True)  # 添加网格

# 显示图形
plt.show()
