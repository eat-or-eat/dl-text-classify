import torch
import torch.nn as nn
from torch.optim import Adam, SGD


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1  # 词嵌入的维度从1开始算
        class_num = config['class_num']
        model_type = config["model_type"]

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == 'lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size)
        elif model_type == 'gru':
            self.encoder = nn.GRU(hidden_size, hidden_size)
        elif model_type == 'rnn':
            self.encoder = nn.RNN(hidden_size, hidden_size)
        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = nn.functional.cross_entropy  # 多分类任务

    def forward(self, x, label=None):
        x = self.embedding(x)  # in:(bs, seq_len)
        x = self.encoder(x)  # in:(bs, seq_len, emb_len)

        if isinstance(x, tuple):  # RNN类型的模型会返回引单元，只取序列结果即可
            x = x[0]
        self.pooling_layer = nn.AvgPool1d(x.shape[1])  # in:(bs, seq_len, emb_len)
        # in:(bs, seq_len, emb_len)->(bs, emb_len, seq_len)->(bs, emb_len, 1)
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        predict = self.classify(x)  # in:(bs, emb_len)
        if label is not None:
            return self.loss(predict, label.squeeze())
        else:
            return predict


# 设置优化器
def select_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
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
