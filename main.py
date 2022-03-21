import numpy as np
from math import log
from math import pow
from numpy import random

def label_uniq_cnt(data):
    """统计data中不同类标签label的个数
    输入：data(list)
    输出：label_uniq_num_cnt(int)
    样本中各个标签的个数{'label1': 2, 'label2': 2, 'label3': 1}
    """
    
    label_uniq_cnt = {}
    for x in data:
        label = x[len(x) - 1] # 取得该样本的标签
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0 # 若标签不在标签字典内，初始化
        label_uniq_cnt[label] = label_uniq_cnt[label] + 1 # 若标签在标签字典内，则个数加一
        
    return label_uniq_cnt

def cal_gini_index(data):
    """计算给定数据集的Gini指数
    输入：data(list)
    输出：gini(float)
    """

    if len(data) == 0:
        return 0
    label_counts = label_uniq_cnt(data)

    gini = 0
    for label in label_counts:
        gini = gini + pow(label_counts[label], 2)
    gini = 1 - float(gini) / pow(len(data), 2)
    
    return gini

class node:
    """类： 树的节点
    """
    def __init__(self,fea=-1,value=None,results=None,right=None,left=None):
        self.fea = fea # 根节点（用于切分数据集的特征）的列索引值
        self.value = value # 待切分的特征 的具体值
        self.results = results # 叶子节点所属的类别
        self.right = right # 该node的右子树
        self.left = left # 该node的左子树

def split_tree(data,fea,value):
    """根据fea对应特征中的特征值value 将 data 划分为左右子树
    输入：data（list）,fea(int),value(float)
    """

    set_1 = []
    set_2 = []
    for item in data:
        if item[fea] >= value:
            set_1.append(item)
        else:
            set_2.append(item)
            
    return (set_1,set_2)

def build_tree(data):
    """构建决策树，函数返回改决策树的根节点
    输入：训练样本data
    输出：树的根节点node
    """

    if len(data) == 0:
        return node()
    #     计算当前的Gini指数
    originGini = cal_gini_index(data)
    bestGain = 0.0  # 初始化最大信息增益
    bestCriteria = None  # 存储最佳切分属性以及最佳切分点
    bestSets = None  # 存储切分后的两个数据集

    feature_num = len(data[0]) - 1  # 样本特征个数
    #     找到根节点
    for fea in range(0, feature_num):
        #         属性节点所有可能取值
        feature_values = {}
        for item in data:
            feature_values[item[fea]] = 1  # 存储特征fea所有可能的取值
        #         计算不同属性节点划分数据后的gini指数
        for value in feature_values.keys():
            (set_1, set_2) = split_tree(data, fea, value)  # 划分左右子树
            theGini = float(len(set_1) * cal_gini_index(set_1) + len(set_2) * cal_gini_index(set_2)) / len(data)
            gain = originGini - theGini  # 计算信息增益

            if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea, value)  # 存储最佳切分节点
                bestSets = (set_1, set_2)
    #    判断划分是否结束
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
    else:
        return node(results=label_uniq_cnt(data))  # 返回当前的类别标签作为最终的类别标签

def choose_samples(data, k):
    """随机选取样本及特征
    输入：数据，特征
    输出：随机的样本，随机的特征
    """
 
    m, n = np.shape(data)
    #     选择出的k个特征的index
    feature = []
    for j in range(k):
        feature.append(random.randint(0, n-2))  # 最后一列是标签 n-1列
    samples_index = []
    for i in range(m):
        samples_index.append(random.randint(0, m-1))
    #     将m个样本的k个特征，组成数据集data_samples
    data_samples = []
    for i in range(m):
        new_sample = []
        for fea in feature:
            new_sample.append(data[samples_index[i]][fea])  # 特征
        # data_index = samples_index[i]
        new_sample.append(data[samples_index[i]][-1])  # 标签
        data_samples.append(new_sample)  # 将k个特征的样本添加到data_samples中
        
    return data_samples, feature

def random_forest_training(train_data, trees_num):
    """
    输入：训练数据，决策树个数
    输出：每棵树 ， 每棵树的节点
    """

    trees_results = []
    trees_feature = []

    n = np.shape(train_data)[1]  # 样本维度，特征数
    if n > 2:
        k = int(log(n-1, 2)) + 1  # 设置特征个数, k 取为 log2n
    else:
        k = 1
    #     构建决策树（有放回的选择样本，无放回的选择特征）
    for i in range(trees_num):
        data_samples, feature = choose_samples(train_data, k)  # 随机选择样本，特征
        tree = build_tree(data_samples)  # 用随机选择的样本构建一棵决策树
        trees_results.append(tree)  # 保存决策树到森林
        trees_feature.append(feature)  # 保存该决策树用到的特征

    return trees_results, trees_feature

def predict(sample, tree):
    """对每一个样本sample进行预测
    函数 predict 利用训练好的 CART 分类树模型 tree 对样本 sample 进行预测。
    当只有树根时，直接返回树根的类标签。
    若此时有左右子树，则根据指定的特征 fea 处的值进行比较，选择左右子树，直到找到最终的标签。
    
    input:  sample(list):需要预测的样本
            tree(类):构建好的分类树
    output: tree.results:所属的类别
    """

    # 1、只是树根
    if tree.results != None:
        return tree.results
    else:
    # 2、有左右子树
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
            
        return predict(sample, branch)

# 测试
from sklearn.datasets import load_iris

iris = load_iris() #载入鸢尾花数据集
train_data = iris
t_data = iris.data.tolist()

trees_results,trees_feature = random_forest_training(t_data,200)
print(trees_results)
print(trees_feature)

sample = t_data[random.randint(0,len(t_data))]
print(sample)
pre = predict(sample,trees_results[random.randint(0,2)])
print(pre)