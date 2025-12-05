"""
生成训练数据脚本
生成2000条正常的时间序列数据用于训练异常检测模型
"""
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta


def generate_normal_time_series(n_samples=2000, n_features=3, noise_level=0.1, seed=42):
    """
    生成正常的时间序列数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        noise_level: 噪声水平
        seed: 随机种子
        
    Returns:
        DataFrame: 包含时间序列数据的DataFrame
    """
    np.random.seed(seed)
    
    # 生成时间戳
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # 生成基础趋势和周期性模式
    t = np.arange(n_samples)
    
    # 为每个特征生成不同的模式
    data = {}
    data['timestamp'] = timestamps
    
    for i in range(n_features):
        # 基础趋势（线性或轻微非线性）
        trend = 10 + 0.01 * t + 0.0001 * t ** 2
        
        # 周期性模式（日周期、周周期等）
        daily_cycle = 2 * np.sin(2 * np.pi * t / 24)  # 24小时周期
        weekly_cycle = 1.5 * np.sin(2 * np.pi * t / (24 * 7))  # 周周期
        
        # 季节性趋势
        seasonal = 3 * np.sin(2 * np.pi * t / (24 * 30))  # 月周期
        
        # 组合所有模式
        base_signal = trend + daily_cycle + weekly_cycle + seasonal
        
        # 添加随机噪声
        noise = np.random.normal(0, noise_level * np.std(base_signal), n_samples)
        
        # 添加轻微的自相关（AR过程）
        ar_component = np.zeros(n_samples)
        ar_component[0] = np.random.normal(0, 1)
        for j in range(1, n_samples):
            ar_component[j] = 0.3 * ar_component[j-1] + np.random.normal(0, 0.5)
        
        # 最终特征值
        feature_values = base_signal + noise + 0.5 * ar_component
        
        # 确保值为正（根据实际需求调整）
        feature_values = feature_values - np.min(feature_values) + 1
        
        data[f'feature_{i+1}'] = feature_values
    
    df = pd.DataFrame(data)
    return df


def main():
    parser = argparse.ArgumentParser(description='生成训练数据')
    parser.add_argument('--output', type=str, default='training_data.csv', 
                       help='输出文件路径（默认：training_data.csv）')
    parser.add_argument('--samples', type=int, default=2000, 
                       help='样本数量（默认：2000）')
    parser.add_argument('--features', type=int, default=3, 
                       help='特征数量（默认：3）')
    parser.add_argument('--noise', type=float, default=0.1, 
                       help='噪声水平（默认：0.1）')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子（默认：42）')
    parser.add_argument('--no-timestamp', action='store_true', 
                       help='不包含时间戳列（仅数值列）')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("生成训练数据")
    print("=" * 50)
    print(f"样本数量: {args.samples}")
    print(f"特征数量: {args.features}")
    print(f"噪声水平: {args.noise}")
    print(f"随机种子: {args.seed}")
    
    # 生成数据
    df = generate_normal_time_series(
        n_samples=args.samples,
        n_features=args.features,
        noise_level=args.noise,
        seed=args.seed
    )
    
    # 如果不需要时间戳，只保留数值列
    if args.no_timestamp:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
    
    # 保存数据
    df.to_csv(args.output, index=False)
    print(f"\n数据已保存到: {args.output}")
    print(f"数据形状: {df.shape}")
    print(f"\n前5行数据预览:")
    print(df.head())
    print(f"\n数据统计信息:")
    print(df.describe())


if __name__ == '__main__':
    main()

