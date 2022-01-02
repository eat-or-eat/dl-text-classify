import torch
from torch.utils.data import DataLoader, random_split

"""
数据加载脚本，数据28分割成测试训练集
流程：
读取文件->
解析数据成[input_id, label_id]对的类数据->
random_split切分数据集->
传入DataLoader
"""


def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()  # 去除行尾换行符
            vocab_dict[token] = index + 1  # 还有padding的位置，让出0来
    return vocab_dict


class DataGenerator:
    '''
    数据生成类：文本->longtensor数据对
    '''

    def __init__(self, config):
        self.config = config
        self.path = config['data_path']
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.load_data()

    def load_data(self):
        self.data = []
        self.sentences = []
        self.labels = []
        with open(self.path, encoding='utf8') as f:
            # 获取句子和标签
            for line in f:
                line_l = line.split(',')
                sentence = line_l[2]
                self.sentences.append(sentence)
                self.labels.append(line_l[1])
            # 整理成数字序列
            self.labels = self.labels[1:]
            self.sentences = self.sentences[1:]
            label_set = set(self.labels)
            self.config['class_num'] = len(label_set)
            self.label2index = dict((label, index) for index, label in enumerate(label_set))
            self.labels = [self.label2index[label] for label in self.labels]
            for i in range(len(self.labels)):
                input_id = torch.LongTensor(self.encode_sentence(self.sentences[i]))
                label_id = torch.LongTensor([self.labels[i]])
                self.data.append([input_id, label_id])

    def encode_sentence(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用DataLoader类封装数据
def load_dataset(config, shuffle=True):
    dg = DataGenerator(config)
    train_size = int(0.8 * len(dg))
    test_size = len(dg) - train_size
    train_dataset, test_dataset = random_split(dg, [train_size, test_size])
    train_dataset = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=shuffle)
    test_dataset = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=shuffle)
    return train_dataset, test_dataset


if __name__ == '__main__':
    import sys

    sys.path.append("..")
    from config import config

    config['vocab_path'] = '../bert-base-chinese/vocab.txt'
    config['data_path'] = '../data/tianchi_data.csv'

    # 数据生成类测试样例
    # DG = DataGenerator(config)

    # 检查数据样式
    train_dataset, test_dataset = load_dataset(config)
    for dataset in [train_dataset, test_dataset]:
        for x, y in dataset:
            break
        print(len(dataset))
        print(x.shape, y.shape)
