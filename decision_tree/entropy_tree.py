import pandas as pd
from math import log2


def creat_dataset():
    """
    创建并返回西瓜数据集3.0
    """
    data = [
        # 编号, 色泽, 根蒂, 敲声, 纹理, 脐部, 触感, 密度, 含糖率, 好瓜
        [1, '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是'],
        [2, '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是'],
        [3, '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '是'],
        [4, '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '是'],
        [5, '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '是'],
        [6, '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '是'],
        [7, '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '是'],
        [8, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '是'],
        [9, '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '否'],
        [10, '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '否'],
        [11, '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '否'],
        [12, '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '否'],
        [13, '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '否'],
        [14, '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '否'],
        [15, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '否'],
        [16, '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '否'],
        [17, '青绿', '蜷缩', '沉闷', '稍糊', '凹陷', '硬滑', 0.719, 0.103, '否']
    ]

    # 定义列名
    columns = ['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']
    df = pd.DataFrame(data, columns=columns)

    # 基础算法中先只使用离散属性'
    df = df.drop(columns=['编号', '密度', '含糖率'])

    return df


# 创建数据集
df = creat_dataset()


def calc_entropy(dataset):
    """
    计算给定数据集的信息熵
    :param dataset: pandas DataFrame，最后一列是标签
    :return: float，信息熵的值
    """
    num_entries = len(dataset)
    if num_entries == 0:
        return 0

    # 统计每个类别的数量
    label_counts = dataset.iloc[:, -1].value_counts()

    entropy = 0.0
    for count in label_counts:
        # 计算该类别的概率
        prob = count / num_entries
        # 累加熵值
        entropy -= prob * log2(prob)

    return entropy


def split_dataset(dataset, feature_name, value):
    """
    根据指定特征和值分割数据集
    :param dataset: pandas DataFrame，待分割的数据集
    :param feature_name: string，用于分割的特征的列名
    :param value: string，特征的具体值
    :return: pandas DataFrame，分割后的新数据集（已移除该特征列）
    """
    # 筛选出 feature_name 列的值等于 value的所有行
    filtered_df = dataset[dataset[feature_name] == value]

    # 删除已经用过的 feature_name 列，并返回结果
    result = filtered_df.drop(columns=[feature_name]).copy()

    return result


def choose_best_feature_to_split(dataset):
    """
    选择最优的特征进行划分
    :param dataset: pandas DataFrame，待划分的数据集
    :return: string，最优特征的列名
    """
    # 获取特征数量
    num_features = len(dataset.columns) - 1
    # 计算整个数据集的原始信息熵
    base_entropy = calc_entropy(dataset)

    # 初始化最优信息增益和最优特征
    best_info_gain = 0.0
    beat_feature_name = None

    # 遍历所有特征
    for i in range(num_features):
        feature_name = dataset.columns[i]
        # 获取该特征下所有的唯一值
        unique_values = set(dataset[feature_name])

        # 计算该特征划分下的加权信息熵
        new_entropy = 0.0
        for value in unique_values:
            # 分割数据集
            sub_dataset = split_dataset(dataset, feature_name, value)
            # 计算子集的概率
            prob = len(sub_dataset) / len(dataset)
            # 累积加权熵
            new_entropy += prob * calc_entropy(sub_dataset)

        # 计算信息增益
        info_gain = base_entropy - new_entropy

        # 比较并更新最优信息增益和最优特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            beat_feature_name = feature_name

    return beat_feature_name


def majority_cnt(class_list):
    """
    采用多数表决的方法决定叶子结点的分类
    :param class_list: list，类别标签列表
    :return: string，出现最多的类别
    """
    major_class = class_list.value_counts().index[0]
    return major_class


def create_tree(dataset):
    """
    递归函数，创建决策树
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

    # ---递归过程---
    # 选择最优划分特征
    best_feature_name = choose_best_feature_to_split(dataset)

    # 创建当前结点
    my_tree = {best_feature_name: {}}

    # 得到最优特征的所有唯一值
    unique_vals = set(dataset[best_feature_name])

    # 遍历所有唯一值，为每个值创建一个分支
    for value in unique_vals:
        # 分割数据集
        sub_dataset = split_dataset(dataset, best_feature_name, value)
        # 递归调用 create_tree ，并将返回的子树或标签作为分支内容
        my_tree[best_feature_name][value] = create_tree(sub_dataset)

    return my_tree


# --- 最终执行与生成决策树 ---
print("\n--- 开始生成决策树 ---")
my_decision_tree = create_tree(df)

print("\n生成的决策树（字典结构）为:")
import pprint
pprint.pprint(my_decision_tree)

