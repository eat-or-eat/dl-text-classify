import os
import logging
import numpy as np
import torch
import time
import pandas as pd

from config import config
from src.loader import load_dataset
from src.model import Model, select_optimizer
from src.evaluater import EvalData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    start_time = time.time()
    train_dataset, test_dataset = load_dataset(config)  # 加载数据集
    model = Model(config)  # 加载模型
    cuda_flag = torch.cuda.is_available()  # 是否加载到GPU
    if cuda_flag:
        logger.info('GPU YES,training with GPU')
        model = model.cuda()
    optimizer = select_optimizer(config, model)  # 加载优化器
    evaluator = EvalData(config, model, logger)  # 加载验证类
    train_f1s, eval_f1s, losses, best_f1 = [], [], [], 0
    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info('epoch %s start' % epoch)
        train_loss = []
        for index, data in enumerate(train_dataset):
            if cuda_flag:
                data = [datum.cuda() for datum in data]
            optimizer.zero_grad()
            input_ids, labels = data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_dataset) / 3) == 0:
                # logger.info('batch loss %f' % loss)
                pass
        train_loss = np.mean(train_loss)
        logger.info('第%d轮模型loss: %f' % (epoch, np.mean(train_loss)))
        losses.append(train_loss)
        train_f1 = evaluator.eval(epoch, train_dataset)
        train_f1s.append(train_f1)
        eval_f1 = evaluator.eval(epoch, test_dataset)
        eval_f1s.append(eval_f1)
        # 这个保存模型还可以优化，这里只是保存了这一轮参数里面最好的
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            model_path = os.path.join(config['model_path'] + config['model_type'])
            # 创建保存模型的目录
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            pth_path = os.path.join(config['model_path'] + config['model_type'], 'best.pth')
            torch.save(model.state_dict(), pth_path)
    evaluator.plot_and_save(epoch, train_f1s, eval_f1s, losses)
    used_time = time.time() - start_time

    result_dict = {'train_f1s': train_f1s,
                   'eval_f1s': eval_f1s,
                   'losses': losses,
                   'best_f1': best_f1,
                   'used_time': used_time
                   }
    return result_dict


if __name__ == '__main__':
    # 测试用例
    # for model in ['rnn', 'lstm', 'gru']:
    #     config['model_type'] = model
    #     print('当前模型:', model)
    #     result_dict = main(config)

    model_results = []
    # 'lstm', 'gru', 'rnn', 'cnn', 'gated_cnn', 'rcnn', 'bert', 'bert_lstm', 'bert_mid'
    for model in ['cnn']:
        results = []
        config['model_type'] = model
        for lr in [1e-2]:  # 1e-2, 1e-3, 1e-4
            config['learning_rate'] = lr
            for hidden_size in [32]:  # 32, 64, 128, 256
                config['hidden_size'] = hidden_size
                for opt in ['adam']:  # 'adam', 'sgd'
                    config['optimizer'] = opt
                    print('当前配置:\n', config)
                    result_dict = main(config)
                    result_temp = [config['model_type'],
                                   config['learning_rate'],
                                   config['hidden_size'],
                                   config['optimizer'],
                                   result_dict['used_time'],
                                   result_dict['best_f1']]
                    results.append(result_temp)
        model_result = sorted(results, reverse=True, key=lambda x: x[-1])
        model_results.extend(model_result)
    df = pd.DataFrame(model_results)
    df.to_csv(config['model_path'] + 'models_comparison.csv',
              header=['model_type',
                      'learning_rate',
                      'hidden_size',
                      'optimizer',
                      'used_time',
                      'best_f1'])
