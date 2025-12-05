"""
异常检测模块
加载模型并进行异常检测
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
from model import Autoencoder
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """异常检测器类"""
    
    def __init__(self, model_path, threshold_path, scaler_path, config_path=None, device=None):
        """
        初始化异常检测器
        
        Args:
            model_path: 模型文件路径
            threshold_path: 阈值文件路径
            scaler_path: 标准化器文件路径
            config_path: 模型配置文件路径（可选，默认从model_path推断）
            device: 设备 (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载模型配置和模型
        self.model = self._load_model(model_path, config_path)
        
        # 加载阈值
        self.threshold = self._load_threshold(threshold_path)
        
        # 加载标准化器
        self.scaler = self._load_scaler(scaler_path)
    
    def _load_model(self, model_path, config_path=None):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 先加载state_dict以检查结构
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 从state_dict完全推断模型结构（更可靠）
        encoder_weight_keys = [k for k in state_dict.keys() if 'encoder' in k and 'weight' in k]
        decoder_weight_keys = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k]
        
        encoder_weight_keys.sort(key=lambda x: int(x.split('.')[1]))
        decoder_weight_keys.sort(key=lambda x: int(x.split('.')[1]))
        
        if not encoder_weight_keys or not decoder_weight_keys:
            raise ValueError("无法从模型文件推断结构，state_dict格式不正确")
        
        # 从encoder的第一层获取input_dim
        first_encoder_key = encoder_weight_keys[0]
        input_dim = state_dict[first_encoder_key].shape[1]
        
        # 从encoder的最后一层获取encoding_dim
        last_encoder_key = encoder_weight_keys[-1]
        encoding_dim = state_dict[last_encoder_key].shape[0]
        
        # 推断hidden_dims（encoder中间层的输出维度）
        hidden_dims = []
        if len(encoder_weight_keys) > 1:
            for key in encoder_weight_keys[:-1]:
                hidden_dims.append(state_dict[key].shape[0])
        
        # 验证decoder结构是否匹配
        # decoder的第一层应该从encoding_dim开始
        first_decoder_key = decoder_weight_keys[0]
        if state_dict[first_decoder_key].shape[1] != encoding_dim:
            print(f"警告: decoder第一层输入维度 ({state_dict[first_decoder_key].shape[1]}) 与编码层维度 ({encoding_dim}) 不匹配")
        
        # decoder的最后一层应该输出input_dim
        last_decoder_key = decoder_weight_keys[-1]
        if state_dict[last_decoder_key].shape[0] != input_dim:
            print(f"警告: decoder最后一层输出维度 ({state_dict[last_decoder_key].shape[0]}) 与输入维度 ({input_dim}) 不匹配")
        
        # 创建模型
        model = Autoencoder(input_dim, encoding_dim, hidden_dims).to(self.device)
        
        # 尝试加载state_dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # 如果严格加载失败，尝试非严格加载
            print(f"警告: 严格加载模型失败，尝试非严格加载")
            print(f"错误信息: {str(e)}")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"缺失的键: {missing_keys}")
                if unexpected_keys:
                    print(f"意外的键: {unexpected_keys}")
                print("模型已使用非严格模式加载")
            except Exception as e2:
                raise RuntimeError(f"无法加载模型: {str(e2)}")
        
        model.eval()
        
        return model
    
    def _load_threshold(self, threshold_path):
        """加载阈值"""
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"阈值文件不存在: {threshold_path}")
        
        with open(threshold_path, 'r', encoding='utf-8') as f:
            threshold_info = json.load(f)
        
        return threshold_info['threshold']
    
    def _load_scaler(self, scaler_path):
        """加载标准化器"""
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return scaler
    
    def load_data(self, file_path):
        """
        加载待检测数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            data: 标准化后的数据
            original_df: 原始数据DataFrame
            numeric_cols: 数值列名列表
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext == '.txt':
            try:
                df = pd.read_csv(file_path, sep='\t')
            except:
                try:
                    df = pd.read_csv(file_path, sep=',')
                except:
                    df = pd.read_csv(file_path, sep=' ')
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .csv 和 .txt")
        
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("数据文件中没有找到数值列")
        
        data = df[numeric_cols].values
        
        # 标准化
        data_scaled = self.scaler.transform(data)
        
        return data_scaled, df, numeric_cols
    
    def detect(self, data):
        """
        检测异常
        
        Args:
            data: 待检测数据 (numpy array, 已标准化)
            
        Returns:
            results: 检测结果列表，每个元素为 {
                'index': 行索引,
                'reconstruction_error': 重构误差,
                'is_anomaly': 是否异常
            }
        """
        self.model.eval()
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        results = []
        
        with torch.no_grad():
            reconstructed = self.model(data_tensor)
            reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            
            for idx, error in enumerate(reconstruction_errors):
                is_anomaly = error > self.threshold
                results.append({
                    'index': idx,
                    'reconstruction_error': float(error),
                    'is_anomaly': bool(is_anomaly)
                })
        
        return results
    
    def detect_from_file(self, file_path):
        """
        从文件加载数据并检测异常
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            results: 检测结果列表
            original_df: 原始数据DataFrame
        """
        data, original_df, _ = self.load_data(file_path)
        results = self.detect(data)
        return results, original_df


def main():
    """命令行测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='异常检测')
    parser.add_argument('--data', type=str, required=True, help='待检测数据文件路径')
    parser.add_argument('--model', type=str, default='models/model.pth', help='模型文件路径')
    parser.add_argument('--threshold', type=str, default='models/threshold.json', help='阈值文件路径')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl', help='标准化器文件路径')
    parser.add_argument('--output', type=str, default=None, help='结果输出文件路径（可选）')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("开始异常检测")
    print("=" * 50)
    
    # 初始化检测器
    detector = AnomalyDetector(args.model, args.threshold, args.scaler)
    
    # 检测
    print(f"\n加载数据: {args.data}")
    results, original_df = detector.detect_from_file(args.data)
    
    # 统计结果
    total = len(results)
    anomalies = sum(1 for r in results if r['is_anomaly'])
    
    print(f"\n检测完成！")
    print(f"总样本数: {total}")
    print(f"异常样本数: {anomalies}")
    print(f"异常比例: {anomalies/total*100:.2f}%")
    
    # 显示前10个异常
    anomaly_results = [r for r in results if r['is_anomaly']]
    if anomaly_results:
        print(f"\n前10个异常样本:")
        for i, r in enumerate(anomaly_results[:10]):
            print(f"  行 {r['index']}: 重构误差 = {r['reconstruction_error']:.6f}")
    
    # 保存结果
    if args.output:
        output_df = original_df.copy()
        output_df['reconstruction_error'] = [r['reconstruction_error'] for r in results]
        output_df['is_anomaly'] = [r['is_anomaly'] for r in results]
        output_df.to_csv(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()

