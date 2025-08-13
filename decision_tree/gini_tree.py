import pandas as pd
import pprint


def create_full_dataset():
    """
    创建西瓜数据集2.0
    """
    data = [
        # 编号, 色泽, 根蒂, 敲声, 纹理, 脐部, 触感, 好瓜
        [1, '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        [2, '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        [3, '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        [4, '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        [5, '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        [6, '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
        [7, '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
        [8, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        [9, '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
        [10, '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        [11, '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        [12, '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
        [13, '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
        [14, '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        [15, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
        [16, '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        [17, '青绿', '蜷缩', '沉闷', '稍糊', '凹陷', '硬滑', '否']
    ]
    columns = ['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    # 创建完整的 DataFrame，去掉无关的'编号'列
    df = pd.DataFrame(data, columns=columns).drop(columns=['编号'])
    return df


# 准备全量数据
full_df = create_full_dataset()

# 划分训练集和验证集
train_indices = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
validation_indices = [3, 4, 7, 8, 10, 11, 12]

train_df = full_df.loc[train_indices].reset_index(drop=True)
validation_df = full_df.loc[validation_indices].reset_index(drop=True)


def split_dataset(dataset, feature_name, value):
    """
    根据指定特征和值分割数据集
    :param dataset: pandas DataFrame，待分割的数据集
    :param feature_name: string，用于分割的特征的列名
    :param value: string，特征的具体值
    :return: pandas DataFrame，分割后的新数据集（已移除该特征列）
    """
    filtered_df = dataset[dataset[feature_name] == value]
    result = filtered_df.drop(columns=[feature_name]).copy()
    return result


def majority_cnt(class_list):
    """
    采用多数表决的方法决定叶子结点的分类
    :param class_list: list，类别标签列表
    :return: string，出现最多的类别
    """
    major_class = class_list.value_counts().index[0]
    return major_class


def calc_gini_index(dataset):
    """
    计算给定数据集的基尼系数
    :param dataset: pandas DataFrame
    :return: float，基尼指数的值
    """
    num_entries = len(dataset)
    if num_entries == 0:
        return 0    # 空集的基尼指数定义为0

    # 统计每个类别的数量
    label_counts = dataset.iloc[:, -1].value_counts()

    gini = 1.0
    for count in label_counts:
        # 计算该类别的概率
        prob = count / num_entries
        # 根据公式累减
        gini -= prob * prob

    return gini


def choose_best_feature_gini(dataset):
    """
    使用基尼指数选择最优的特征进行划分
    :param dataset: pandas DataFrame，待划分的数据集
    :return: string，最优特征的列名
    """
    # 获取特征数量
    num_features = len(dataset.columns) - 1
    # 初始化最优基尼指数和最优特征
    best_gini_index = float('inf')
    best_feature_name = None

    # 遍历所有特征
    for i in range(num_features):
        feature_values = dataset.columns[i]
        # 获取该特征下所有的唯一值
        unique_vals = set(dataset[feature_values])

        # 计算该特征划分下的加权基尼指数
        current_gini_index = 0.0
        for value in unique_vals:
            # 分割数据集
            sub_dataset = split_dataset(dataset, feature_values, value)
            # 计算子集的权重
            prob = len(sub_dataset) / len(dataset)
            # 累加加权基尼指数
            current_gini_index += prob * calc_gini_index(sub_dataset)

        # 比较并更新最优基尼指数和最优特征
        if current_gini_index < best_gini_index:
            best_gini_index = current_gini_index
            best_feature_name = feature_values

    return best_feature_name


def create_tree_unpruned(dataset):
    """
    递归函数，创建一棵完整的、未剪枝的决策树
    :param dataset: pandas DataFrame，当前结点的数据集
    :return: dict of string，嵌套字典表示的决策树，或叶子结点的类标签
    """
    class_list = dataset.iloc[:, -1]

    # 终止条件1：数据集中所有实例都属于同一类别
    if len(class_list.unique()) == 1:
        return class_list.iloc[0]

    # 终止条件2：所有特征已经用完，但类别还不唯一
    if len(dataset.columns) == 1:
        return majority_cnt(class_list)

    # --- 递归过程 ---
    # 选择最优划分特征
    best_feature_name = choose_best_feature_gini(dataset)

    # 创建当前结点
    my_tree = {best_feature_name: {}}

    # 得到最优特征的所有唯一值
    unique_vals = set(dataset[best_feature_name])

    # 遍历所有唯一值，为每个值创建分支并递归
    for value in unique_vals:
        sub_dataset = split_dataset(dataset, best_feature_name, value)
        my_tree[best_feature_name][value] = create_tree_unpruned(sub_dataset)

    return my_tree


# --- 生成并打印未剪枝的树 ---
print("\n--- 1. 开始生成未剪枝决策树 ---")
unpruned_tree = create_tree_unpruned(train_df)

print("\n生成的未剪枝决策树（字典结构）为:")
pprint.pprint(unpruned_tree)


def predict(tree, data_point):
    """
    使用决策树对单个数据点进行预测
    :param tree: dict of string，已经训练好的决策树
    :param data_point: pandas Series，一行数据
    :return: string，预测的类别标签
    """
    # 如果 tree 不是字典，说明已达到叶子结点，直接返回结果
    if not isinstance(tree, dict):
        return tree

    # 获取当前结点的决策特征名
    feature_name = list(tree.keys())[0]
    # 获取该特征下的所有分支（子树）
    sub_tree = tree[feature_name]
    # 获取数据点在该特征上的具体取值
    feature_value = data_point[feature_name]

    # 如果数据点的特征值存在于分支中，则递归进入该分支
    if feature_value in sub_tree:
        return predict(sub_tree[feature_value], data_point)
    else:
        # 如果训练中未见过该特征值，无法判断，返回一个默认值
        return None


def test_accuracy(tree, test_dataset):
    """
    在测试集上评估决策树的准确率
    :param tree: dict of string，决策树
    :param test_dataset: pandas DataFrame，测试/验证数据集
    :return: float，准确率
    """
    if len(test_dataset) == 0:
        return 0

    correct_predictions = 0
    # 遍历验证集的每一行
    for index, row in test_dataset.iterrows():
        # 得到预测结果
        prediction = predict(tree, row)
        # 与真实标签比较
        if prediction == row.iloc[-1]:
            correct_predictions += 1

    return correct_predictions / len(test_dataset)


# --- 评估未剪枝树的性能 ---
unpruned_accuracy = test_accuracy(unpruned_tree, validation_df)
print(f"\n预剪枝决策树在 7 条验证集上的准确率为: {unpruned_accuracy:.2%}")


def create_tree_preprune(train_dataset, validation_dataset):
    """
    递归函数，创建预剪枝决策树
    :param train_dataset: pandas DataFrame，当前结点的训练数据集
    :param validation_dataset: pandas DataFrame，当前结点的验证数据集
    :return: dict of string，经过预剪枝的决策树或叶子结点标签
    """
    class_list = train_dataset.iloc[:, -1]

    # 终止条件1：训练集已经纯净
    if len(class_list.unique()) == 1:
        return class_list.iloc[0]

    # ---预剪枝核心逻辑---
    # 计算不划分时的准确率
    majority_class = majority_cnt(class_list)
    accuracy_no_split = test_accuracy(majority_class, validation_dataset)

    # 终止条件2：如果所有特征用完，也无法划分，返回多数类
    if len(train_dataset.columns) == 1:
        return majority_class

    # 计算划分后的准确率
    best_feature_name = choose_best_feature_gini(train_dataset)
    # 临时构建划分后的树，其叶子结点都用训练子集的多数类来标记
    temp_tree = {best_feature_name: {}}
    for value in set(train_dataset[best_feature_name]):
        sub_dataset = split_dataset(train_dataset, best_feature_name, value)
        # 如果划分后子集为空，则用父结点的多数类
        temp_tree[best_feature_name][value] = majority_cnt(sub_dataset.iloc[:, -1])

    accuracy_with_split = test_accuracy(temp_tree, validation_dataset)

    # 比较准确率，决定是否剪枝
    if accuracy_with_split <= accuracy_no_split:
        return majority_class   # 剪枝：准确率没有严格提升，不划分

    # 如果不剪枝，则正常递归创建
    my_tree = {best_feature_name: {}}
    for value in set(train_dataset[best_feature_name]):
        sub_dataset = split_dataset(train_dataset, best_feature_name, value)
        # 验证集也需要进行相应的划分，传递给下一层递归
        sub_validation_dataset = validation_dataset[validation_dataset[best_feature_name] == value].drop(columns=[best_feature_name])

        my_tree[best_feature_name][value] = create_tree_preprune(sub_dataset, sub_validation_dataset)

    return my_tree


# --- 生成并打印预剪枝的树 ---
print("\n--- 2. 开始生成预剪枝决策树 ---")
prepruned_tree = create_tree_preprune(train_df, validation_df)
print("\n生成的预剪枝决策树（字典结构）为:")
pprint.pprint(prepruned_tree)

# --- 评估预剪枝树的性能 ---
prepruned_accuracy = test_accuracy(prepruned_tree, validation_df)
print(f"\n预剪枝决策树在 7 条验证集上的准确率为: {prepruned_accuracy:.2%}")


import copy


def create_tree_postprune(train_dataset, validation_dataset):
    """
    创建后剪枝决策树
    :param train_dataset: 训练集数据
    :param validation_dataset: 验证集数据
    :return: 决策树
    """
    # 基于训练集生成一棵完整的未剪枝树
    full_tree = create_tree_unpruned(train_dataset)
    print("\n--- 3. 开始生成后剪枝决策树 ---")
    print("未剪枝树在验证集上的初始准确率: {:.2%}".format(test_accuracy(full_tree, validation_dataset)))

    # 调用递归函数进行剪枝
    pruned_tree = post_prune(full_tree, train_dataset, validation_dataset)
    return pruned_tree


def post_prune(tree, train_dataset, validation_dataset):
    """
    递归执行后剪枝的函数
    :param tree: 完整的树
    :param train_dataset: 训练集
    :param validation_dataset: 测试集
    :return: 剪枝后的树
    """
    # 如果验证集为空，无法进行剪枝评估，直接返回原树
    if validation_dataset.empty:
        return tree

    # 如果当前结点不是决策结点，直接返回
    if not isinstance(tree, dict):
        return tree

    feature_name = list(tree.keys())[0]
    sub_tree = tree[feature_name]

    # 从下往上，先对子树进行剪枝
    for key, value_subtree in sub_tree.items():
        if isinstance(value_subtree, dict):
            # 准备下一层递归所需的数据
            sub_train_dataset = split_dataset(train_dataset, feature_name, key)
            sub_validation_dataset = split_dataset(validation_dataset, feature_name, key)
            # 递归调用，并用剪枝后的子树替换原来的子树
            sub_tree[key] = post_prune(value_subtree, sub_train_dataset, sub_validation_dataset)

    # 核心逻辑：尝试剪枝当前结点
    # 计算不剪枝的准确率
    acc_no_prune = test_accuracy(tree, validation_dataset)

    # 计算剪枝后的准确率
    # 即用当前结点训练数据中的多数类替换整个子树
    majority_class = majority_cnt(train_dataset.iloc[:, -1])
    acc_prune = test_accuracy(majority_class, validation_dataset)

    # 比较准确率
    if acc_prune >= acc_no_prune:
        # 如果剪枝后的准确率更高或持平，则进行剪枝
        print(f"剪枝节点: '{feature_name}', 准确率从 {acc_no_prune:.2%} 提升至 {acc_prune:.2%}")
        return majority_class
    else:
        # 否则，保留原树
        return tree


# --- 生成并打印后剪枝的树 ---
postpruned_tree = create_tree_postprune(train_df, validation_df)
print("\n生成的后剪枝决策树（字典结构）为:")
pprint.pprint(postpruned_tree)

# --- 评估后剪枝树的性能 ---
postpruned_accuracy = test_accuracy(postpruned_tree, validation_df)
print(f"\n后剪枝决策树在 7 条验证集上的准确率为: {postpruned_accuracy:.2%}")

