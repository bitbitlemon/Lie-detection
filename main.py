import torch
from torch import nn, optim
from models.crnn_model import CNNModel
from data.data_loader import create_data_loader
from utils.trainer import Trainer
from utils.visualization import plot_loss
from config.config import Config

def main():
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_loader = create_data_loader(Config.data_path, Config.batch_size, shuffle=True)
    val_loader = create_data_loader(Config.data_path, Config.batch_size, shuffle=False)

    # 初始化模型
    model = CNNModel(Config.input_size, Config.hidden_size, Config.num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # 初始化训练器
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    # 训练过程
    train_losses = []
    val_losses = []
    for epoch in range(Config.num_epochs):
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate_epoch()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{Config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), Config.model_save_path)

    # 绘制损失曲线
    plot_loss(train_losses, val_losses, Config.loss_curve_path)

if __name__ == '__main__':
    main()
