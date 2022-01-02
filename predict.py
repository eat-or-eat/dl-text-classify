import os
import torch
import json

from config import config
from src.loader import load_vocab
from src.model import Model

"""
模型预测类
"""

def load_json(path):
    with open(path, encoding='utf8') as f:
        result = json.load(f)
    return result


class TextClassify:
    def __init__(self, config, model_type):
        self.config = config
        self.config['model_type'] = model_type
        self.vocab = load_vocab(self.config['vocab_path'])
        self.index2label = load_json(self.config['classify_schema_path'])
        print('字表、schema加载完毕')
        self.model = Model(config)
        self.model.load_state_dict(torch.load(os.path.join(self.config['model_path'],
                                                           self.config['model_type'],
                                                           'best.pth')))
        self.model.eval()
        print('模型加载完毕')

    def encode_sentence(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def predict(self, sentence):
        self.input_id = self.encode_sentence(sentence)
        with torch.no_grad():
            self.prod_result = self.model(torch.LongTensor([self.input_id]))
            result = torch.argmax(self.prod_result, dim=-1).item()
        result = self.index2label[str(result)]
        return result


if __name__ == '__main__':
    tc = TextClassify(config, 'cnn')

    # 测试用例
    # sentence = '感冒鼻子干有什么方法能快速解决'
    # result = tc.predict(sentence)
    # print(result)

    sentences = ['支气管炎治疗用的雾化药含有激素吗',  # 支气管炎
                 '得过一次支原体肺炎会复发吗',  # 支原体肺炎
                 '替硝唑片的保质期有多久',  # 肺炎
                 '肺结核吃药期间偶尔的咳嗽几下正常吗']  # 肺结核
    print('test for sentece:')
    for sentence in sentences:
        result = tc.predict(sentence)
        print('原句：%s， 预测结果：%s' % (sentence, result))

    sentence = '感冒鼻子干有什么方法能快速解决'  # 上呼吸道感染
    print('test for short sentence')
    for i in range(len(sentence)):
        short_sentence = sentence[:i]
        result = tc.predict(short_sentence)
        print('原句：%s， 预测结果：%s' % (short_sentence, result))
