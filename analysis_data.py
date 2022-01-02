import os
import matplotlib.pyplot as plt

from collections import defaultdict

"""
该脚本用于分析数据
"""
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 路径配置
data_path = './data/tianchi_data.csv'


def load_data(data_path):
    """
    从语料路径加载语料数据，返回corpus
    :param data_path: 语料相对路径
    :return: 返回句子长度数量列表对，标签数量列表对
    """
    data_len, Classes = defaultdict(int), defaultdict(int)
    with open(data_path, encoding='utf8') as f:
        for line in f:

            line_l = line.split(',')
            if line_l[0] != 'id':
                Classes[line_l[1]] += 1
                data_len[len(line_l[2])] += 1
                data_len[len(line_l[3])] += 1
    data_len = sorted(data_len.items(), key=lambda x: x[0])
    Classes = sorted(Classes.items(), key=lambda x: x[1])
    return data_len, Classes


def plot_bar_and_save(data, title, xlabel, ylabel, fig_len=12, plot_path='./output/analysis/'):
    """
    传入参数画柱状图并保存
    :param data: 数据对
    :param title: 图名称
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param fig_len: 图长度，用于避免xlabel重叠问题
    :param plot_path: 保存相对路径
    """
    fig = plt.figure(figsize=(fig_len, 4))  # 设置画布大小
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x = [i for i in range(1, len(data) + 1)]
    x_label = [x[0] for x in data]
    y = [x[1] for x in data]
    plt.xticks(x, x_label)  # 绘制x刻度标签
    plt.bar(x, y)  # 绘制y刻度标签
    # 创建保存模型的目录
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.savefig(plot_path+title+'.png')
    plt.show()


if __name__ == "__main__":
    data_len, Classes = load_data(data_path)
    plot_bar_and_save(Classes, '类别数量分布图', '类别', '数量')
    plot_bar_and_save(data_len, '长度数量分布图', '长度', '数量')
