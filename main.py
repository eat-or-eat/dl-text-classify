import logging
import numpy as np

import torch

from config import config
from src.loader import load_dataset
from src.model import Model, select_optimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    train_data = load_dataset(config)  # 加载数据集
    model = Model(config)  # 加载模型
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info('GPU YES,training with GPU')
        model = model.cuda()
    optimizer = select_optimizer(config, model)  # 加载优化器

    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info('epoch %s start' % epoch)
        train_loss = []
        for index, data in enumerate(train_data):
            if cuda_flag:
                data = [datum.cuda() for datum in data]
            optimizer.zero_grad()
            input_ids, labels = data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 3) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))


if __name__ == '__main__':
    for model in ['rnn', 'lstm', 'gru']:
        config['model_type'] = model
        print('当前模型:', model)
        main(config)
