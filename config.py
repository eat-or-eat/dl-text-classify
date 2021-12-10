config = {  # 路径系列参数
    'vocab_path': './bert-base-chinese/vocab.txt',
    'data_path': './data/tianchi_data.csv',
    'model_path': './output/model/',
    # 模型系列参数
    'max_length': 30,
    'hidden_size': 128,
    'batch_size': 128,
    'epoch': 10,
    'model_type': 'gru',
    # 优化器系列参数
    'optimizer': 'adam',
    'learning_rate': 1e-3,
}
