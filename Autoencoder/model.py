"""
Autoencoder 模型定义
用于时间序列异常检测
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """自编码器网络结构"""
    
    def __init__(self, input_dim, encoding_dim=None, hidden_dims=None):
        """
        初始化自编码器
        
        Args:
            input_dim: 输入维度（时间序列特征数）
            encoding_dim: 编码层维度，如果为None则使用input_dim的1/4
            hidden_dims: 隐藏层维度列表，如果为None则使用默认结构
        """
        super(Autoencoder, self).__init__()
        
        if encoding_dim is None:
            encoding_dim = max(1, input_dim // 4)
        
        if hidden_dims is None:
            # 默认结构：input_dim -> input_dim//2 -> encoding_dim -> input_dim//2 -> input_dim
            hidden_dim1 = max(1, input_dim // 2)
            hidden_dims = [hidden_dim1]
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # 解码器最后一层不使用激活函数，直接输出原始维度
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            reconstructed: 重构后的输出 [batch_size, input_dim]
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """仅编码"""
        return self.encoder(x)
    
    def decode(self, encoded):
        """仅解码"""
        return self.decoder(encoded)

