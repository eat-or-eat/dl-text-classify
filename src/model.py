import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
定义各种模型
"""
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        embedding_dim = config['embedding_dim']
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1  # 词嵌入的维度从1开始算
        class_num = config['class_num']
        model_type = config['model_type']
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if model_type == 'lstm':
            self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        elif model_type == 'cnn':
            self.encoder = CNN(config)
        elif model_type == 'gated_cnn':
            self.encoder = GatedCNN(config)
        elif model_type == 'rcnn':
            self.encoder = RCNN(config)
        elif model_type == 'bert':
            self.use_bert = True
            self.encoder = Bert(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == 'bert_lstm':
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.bert.config.hidden_size
        elif model_type == 'bert_mid':
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = nn.functional.cross_entropy  # 多分类任务用交叉熵

    def forward(self, x, label=None):
        if self.use_bert:
            # bert_dict:(last_hidden_state, pooler_output)
            # last_hidden_state:(bs, seq_len, hid_size)  常用于获取动态词向量
            # pooler_output:(bs, hid_size)  常用于获取句向量
            x = self.encoder(x)
        else:
            x = self.embedding(x)  # in:(bs, seq_len)
            x = self.encoder(x)  # in:(bs, seq_len, emb_dim)

        if isinstance(x, tuple):  # RNN类型的模型会返回隐含层，只取序列结果即可
            x = x[0]  # x[0]:(bs, seq_len, hid_size)
        self.pooling_layer = nn.AvgPool1d(x.shape[1])
        # 下面这一步,(bs, seq_len, emb_dim)->(bs, emb_dim, seq_len)->(bs, emb_len, 1)
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        predict = self.classify(x)  # in:(bs, emb_dim)
        if label is not None:
            return self.loss(predict, label.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        embedding_dim = config['embedding_dim']
        hidden_size = config['hidden_size']
        kernel_size = config['kernel_size']
        self.cnn = nn.Conv1d(embedding_dim, hidden_size, kernel_size)

    def forward(self, x):
        # 下面的1dconv操作张量变化
        # (bs, seq_len, emb_dim)->
        # (bs, emb_dim, seq_len)->
        # (bs, hid_size, new_seq_len)->
        # (bs, new_seq_len, hid_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)  # sigmoid映射到0-1，更像一个门，通过与少通过与不通过的选择，类似lstm的门控激活函数
        return torch.mul(a, b)


class RCNN(nn.Module):
    def __init__(self, config):
        embedding_dim = config['embedding_dim']
        super(RCNN, self).__init__()
        self.rnn = nn.RNN(embedding_dim, embedding_dim)
        self.cnn = CNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)  # in:(bs, seq_len, emb_dim)
        x = self.cnn(x)  # in:(bs, seq_len, emb_dim)
        return x  # (bs, new_seq_len, hid_size)


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_model_path'])

    def forward(self, x, mid=False):
        x = self.bert(x)
        if mid == True:  # 如果需要中间层就返回中间层的states
            x = x['hidden_states']
        else:
            x = x['last_hidden_state']
        return x


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = Bert(config)
        self.rnn = nn.LSTM(self.bert.bert.config.hidden_size, self.bert.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)
        x, _ = self.rnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = Bert(config)
        self.bert.bert.config.output_hidden_states = True

    def forward(self, x):
        x = self.bert(x, self.bert.bert.config.output_hidden_states)
        x = torch.add(x[-2], x[-1])
        return x


# 设置优化器
def select_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from config import config

    config["class_num"] = 3
    config["vocab_size"] = 20
    config["max_length"] = 5

    model = Model(config)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    label = torch.LongTensor([1, 2])
    print(model(x, label))
