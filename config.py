config = {
    # 路径系列参数
    'vocab_path': './bert-base-chinese/vocab.txt',
    'data_path': './data/tianchi_data.csv',
    'model_path': './output/model/',
    'bert_model_path': './bert-base-chinese',
    # 模型系列参数
    'max_length': 30,
    'embedding_dim': 64,
    'hidden_size': 32,
    'batch_size': 128,
    'epoch': 20,
    'model_type': 'gru',
    "kernel_size": 3,
    # 优化器系列参数
    'optimizer': 'adam',
    'learning_rate': 1e-3,
}
