class Config:
    data_path = 'data/dataset.pth'
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    input_size = 256  # 输入特征维度
    hidden_size = 128  # LSTM 隐层大小
    num_classes = 10   # 类别数量
    model_save_path = 'models/saved_model.pth'
    loss_curve_path = 'outputs/loss_curve.png'
