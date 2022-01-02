import os
import torch
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

"""
模型效果测试类
"""

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class EvalData:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger

    def eval(self, epoch, dataset):
        self.logger.info("第%d轮模型效果测试：" % epoch)
        self.model.eval()
        self.avg_f1 = []
        for index, batch_data in enumerate(dataset):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.get_report(pred_results, labels)
        self.avg_f1 = np.mean(self.avg_f1)
        self.show_result()
        return self.avg_f1

    def get_report(self, preds, labels, avg='macro avg'):
        preds = torch.argmax(preds, dim=-1).cpu().detach().tolist()
        labels = labels.cpu().detach().tolist()
        report_dict = classification_report(preds, labels, output_dict=True, zero_division=0)
        if avg != 'weighted avg':
            F1 = report_dict['macro avg']['f1-score']
        else:
            F1 = report_dict['weighted avg']['f1-score']
        self.avg_f1.append(F1)

    def show_result(self):
        self.logger.info('F1 %f' % self.avg_f1)

    def plot_and_save(self, epoch, train_f1s, eval_f1s, losses):
        best_f1 = max(eval_f1s)
        pic_path = os.path.join(self.config["model_path"] + self.config['model_type'])
        save_flag = False
        names = os.listdir(pic_path)
        for name in names:
            if name.startswith('report'):
                old_f1 = float(name[:-4].split('-')[2])
                if best_f1 > old_f1:
                    os.remove(os.path.join(pic_path, name))
                    save_flag = True
        if len(names) < 2:
            save_flag = True

        if save_flag:
            x = range(epoch)
            fig = plt.figure()
            plt.plot(x, train_f1s, label='train f1')
            plt.plot(x, eval_f1s, label='eval f1')
            plt.plot(x, losses, label='train loss')
            plt.xlabel('epoch')
            plt.ylabel('num')
            plt.title('训练曲线 best eval=%f' % best_f1)
            plt.legend()
            plt.savefig(os.path.join(pic_path, "report-%s-%f.png" % (self.config['model_type'], best_f1)))


