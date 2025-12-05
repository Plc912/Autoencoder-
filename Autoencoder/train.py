"""
训练自编码器模型用于时间序列异常检测
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import argparse
from model import Autoencoder


def load_data(file_path):
    """
    自动识别并加载 CSV 或 TXT 格式的时间序列数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        data: 标准化后的数据 (numpy array)
        scaler: 标准化器对象
        original_data: 原始数据 (DataFrame)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext == '.txt':
        # 尝试不同的分隔符
        try:
            df = pd.read_csv(file_path, sep='\t')
        except:
            try:
                df = pd.read_csv(file_path, sep=',')
            except:
                df = pd.read_csv(file_path, sep=' ')
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .csv 和 .txt")
    
    # 移除非数值列（如时间戳、ID等）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("数据文件中没有找到数值列")
    
    data = df[numeric_cols].values
    
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler, df[numeric_cols]


def train_model(
    data,
    input_dim,
    encoding_dim=None,
    hidden_dims=None,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device=None
):
    """
    训练自编码器模型
    
    Args:
        data: 训练数据 (numpy array)
        input_dim: 输入维度
        encoding_dim: 编码层维度
        hidden_dims: 隐藏层维度列表
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 设备 (cuda/cpu)
        
    Returns:
        model: 训练好的模型
        train_losses: 训练损失列表
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为张量
    data_tensor = torch.FloatTensor(data).to(device)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = Autoencoder(input_dim, encoding_dim, hidden_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    train_losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return model, train_losses


def calculate_threshold(model, data, percentile=95, device=None):
    """
    计算异常检测阈值（基于重构误差的分位数）
    
    Args:
        model: 训练好的模型
        data: 训练数据
        percentile: 分位数（默认95%）
        device: 设备
        
    Returns:
        threshold: 阈值
        reconstruction_errors: 所有样本的重构误差
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    data_tensor = torch.FloatTensor(data).to(device)
    
    with torch.no_grad():
        reconstructed = model(data_tensor)
        reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    threshold = np.percentile(reconstruction_errors, percentile)
    
    return threshold, reconstruction_errors


def save_model_and_threshold(model, threshold, scaler, model_path, threshold_path, scaler_path, config_path=None):
    """
    保存模型、阈值和标准化器
    
    Args:
        model: 训练好的模型
        threshold: 阈值
        scaler: 标准化器
        model_path: 模型保存路径
        threshold_path: 阈值保存路径
        scaler_path: 标准化器保存路径
        config_path: 模型配置保存路径（可选）
    """
    # 确保目录存在
    import os
    for path in [model_path, threshold_path, scaler_path]:
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存模型配置（用于加载时重建模型结构）
    if config_path is None:
        # 确保配置文件也在同一目录
        model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        model_name = os.path.basename(model_path).replace('.pth', '_config.json')
        config_path = os.path.join(model_dir, model_name)
    
    # 确保配置文件目录存在
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    
    # 从模型推断配置
    encoder_layers = list(model.encoder.children())
    linear_layers = [layer for layer in encoder_layers if isinstance(layer, nn.Linear)]
    
    input_dim = linear_layers[0].weight.shape[1]  # 第一层的输入维度
    encoding_dim = linear_layers[-1].weight.shape[0]  # 最后一层（编码层）的输出维度
    
    config = {
        'input_dim': int(input_dim),
        'encoding_dim': int(encoding_dim),
        'hidden_dims': []
    }
    
    # 提取隐藏层维度（除了第一层和最后一层）
    if len(linear_layers) > 2:
        for layer in linear_layers[1:-1]:
            config['hidden_dims'].append(int(layer.out_features))
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"模型配置已保存到: {config_path}")
    
    # 保存阈值
    threshold_info = {
        'threshold': float(threshold),
        'percentile': 95
    }
    with open(threshold_path, 'w', encoding='utf-8') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"阈值已保存到: {threshold_path}")
    
    # 保存标准化器
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"标准化器已保存到: {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description='训练时间序列异常检测模型')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV/TXT)')
    parser.add_argument('--model', type=str, default=None, help='模型保存路径（可选，默认使用版本号）')
    parser.add_argument('--threshold', type=str, default=None, help='阈值保存路径（可选，默认使用版本号）')
    parser.add_argument('--scaler', type=str, default=None, help='标准化器保存路径（可选，默认使用版本号）')
    parser.add_argument('--config', type=str, default=None, help='模型配置保存路径（可选，默认自动生成）')
    parser.add_argument('--version', type=str, default=None, help='模型版本名称（可选，默认使用时间戳）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--encoding_dim', type=int, default=None, help='编码层维度')
    parser.add_argument('--percentile', type=int, default=95, help='阈值计算分位数')
    
    args = parser.parse_args()
    
    # 处理版本号和文件路径
    import os
    from datetime import datetime
    
    if args.version:
        # 使用指定的版本名称
        version_suffix = f"_{args.version}"
    else:
        # 使用时间戳作为版本号
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_suffix = f"_{timestamp}"
    
    # 如果未指定路径，使用版本号生成路径
    if args.model is None:
        args.model = f"models/model{version_suffix}.pth"
    if args.threshold is None:
        args.threshold = f"models/threshold{version_suffix}.json"
    if args.scaler is None:
        args.scaler = f"models/scaler{version_suffix}.pkl"
    
    print("=" * 50)
    print("开始训练时间序列异常检测模型")
    print("=" * 50)
    
    # 加载数据
    print(f"\n加载数据: {args.data}")
    data, scaler, original_data = load_data(args.data)
    input_dim = data.shape[1]
    print(f"数据形状: {data.shape}")
    print(f"输入维度: {input_dim}")
    
    # 训练模型
    print("\n开始训练模型...")
    model, train_losses = train_model(
        data,
        input_dim,
        encoding_dim=args.encoding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # 计算阈值
    print("\n计算异常检测阈值...")
    threshold, reconstruction_errors = calculate_threshold(model, data, percentile=args.percentile)
    print(f"阈值 (第{args.percentile}百分位数): {threshold:.6f}")
    print(f"重构误差范围: [{reconstruction_errors.min():.6f}, {reconstruction_errors.max():.6f}]")
    
    # 保存模型和阈值
    print("\n保存模型和配置...")
    save_model_and_threshold(
        model,
        threshold,
        scaler,
        args.model,
        args.threshold,
        args.scaler,
        args.config
    )
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()

