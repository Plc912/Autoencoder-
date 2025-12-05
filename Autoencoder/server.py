"""
FastMCP 服务器，提供 MCP 工具和 SSE 推送
"""
import os
import json
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP
from detect import AnomalyDetector


# 创建 FastMCP 实例
mcp = FastMCP("autoencoder_anomaly_detection", debug=True, log_level="INFO")

# 全局检测器实例
detector: Optional[AnomalyDetector] = None


def init_detector(
    model_path: str = "models/model.pth",
    threshold_path: str = "models/threshold.json",
    scaler_path: str = "models/scaler.pkl",
    config_path: Optional[str] = None
):
    """初始化检测器"""
    global detector
    try:
        detector = AnomalyDetector(model_path, threshold_path, scaler_path, config_path)
        return True
    except Exception as e:
        raise RuntimeError(f"初始化检测器失败: {str(e)}")


@mcp.tool()
def run_detection(
    file_path: str,
    model_path: str = "models/model.pth",
    threshold_path: str = "models/threshold.json",
    scaler_path: str = "models/scaler.pkl",
    config_path: Optional[str] = None,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    执行时间序列异常检测
    
    参数:
    - file_path: str
        待检测数据文件路径（CSV 或 TXT）
    - model_path: str
        模型文件路径，默认 "model.pth"
    - threshold_path: str
        阈值文件路径，默认 "threshold.json"
    - scaler_path: str
        标准化器文件路径，默认 "scaler.pkl"
    - config_path: Optional[str]
        模型配置文件路径（可选）
    - return_details: bool
        是否返回详细信息（包括每行的检测结果），默认 True
        
    返回:
    - Dict[str, Any]: 检测结果，包含：
        - status: 状态 ("success" 或 "error")
        - total_rows: 总行数
        - anomaly_count: 异常行数
        - anomaly_ratio: 异常比例
        - threshold: 使用的阈值
        - results: 详细结果列表（如果 return_details=True）
        - error: 错误信息（如果 status="error"）
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"文件不存在: {file_path}"
            }
        
        # 初始化检测器（如果尚未初始化或路径改变）
        global detector
        if (detector is None or 
            not hasattr(detector, 'model_path') or 
            detector.model_path != model_path):
            init_detector(model_path, threshold_path, scaler_path, config_path)
            detector.model_path = model_path
        
        # 执行检测
        results, original_df = detector.detect_from_file(file_path)
        
        # 统计信息
        total = len(results)
        anomalies = sum(1 for r in results if r['is_anomaly'])
        
        # 构建返回结果
        response = {
            "status": "success",
            "total_rows": total,
            "anomaly_count": anomalies,
            "anomaly_ratio": anomalies / total if total > 0 else 0,
            "threshold": float(detector.threshold)
        }
        
        # 添加详细结果
        if return_details:
            response["results"] = results
        else:
            # 只返回异常样本的索引
            response["anomaly_indices"] = [
                r['index'] for r in results if r['is_anomaly']
            ]
        
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
def get_model_info(
    model_path: str = "models/model.pth",
    threshold_path: str = "models/threshold.json",
    scaler_path: str = "models/scaler.pkl",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取模型信息
    
    参数:
    - model_path: str
        模型文件路径
    - threshold_path: str
        阈值文件路径
    - scaler_path: str
        标准化器文件路径
    - config_path: Optional[str]
        模型配置文件路径
        
    返回:
    - Dict[str, Any]: 模型信息，包括模型路径、阈值等
    """
    try:
        global detector
        if (detector is None or 
            not hasattr(detector, 'model_path') or 
            detector.model_path != model_path):
            init_detector(model_path, threshold_path, scaler_path, config_path)
            detector.model_path = model_path
        
        return {
            "model_path": model_path,
            "threshold_path": threshold_path,
            "scaler_path": scaler_path,
            "config_path": config_path,
            "threshold": float(detector.threshold),
            "device": str(detector.device)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
def train_model(
    data_file: str,
    model_path: Optional[str] = None,
    threshold_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    config_path: Optional[str] = None,
    version: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    encoding_dim: Optional[int] = None,
    percentile: int = 95
) -> Dict[str, Any]:
    """
    训练自编码器模型用于时间序列异常检测
    
    参数:
    - data_file: str
        训练数据文件路径（CSV 或 TXT）
    - model_path: Optional[str]
        模型保存路径（可选，默认使用版本号自动生成）
    - threshold_path: Optional[str]
        阈值保存路径（可选，默认使用版本号自动生成）
    - scaler_path: Optional[str]
        标准化器保存路径（可选，默认使用版本号自动生成）
    - config_path: Optional[str]
        模型配置保存路径（可选，默认自动生成）
    - version: Optional[str]
        模型版本名称（可选，默认使用时间戳，格式：YYYYMMDD_HHMMSS）
    - epochs: int
        训练轮数，默认 100
    - batch_size: int
        批次大小，默认 32
    - learning_rate: float
        学习率，默认 0.001
    - encoding_dim: Optional[int]
        编码层维度（可选，默认自动计算）
    - percentile: int
        阈值计算分位数，默认 95
        
    返回:
    - Dict[str, Any]: 训练结果信息，包含生成的文件路径
    """
    try:
        # 导入训练模块
        import sys
        from pathlib import Path
        from datetime import datetime
        sys.path.insert(0, str(Path(__file__).parent))
        
        from train import load_data, train_model as train_model_func, calculate_threshold, save_model_and_threshold
        
        # 处理版本号和文件路径
        if version:
            # 使用指定的版本名称
            version_suffix = f"_{version}"
        else:
            # 使用时间戳作为版本号
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_suffix = f"_{timestamp}"
        
        # 如果未指定路径，使用版本号生成路径
        if model_path is None:
            model_path = f"models/model{version_suffix}.pth"
        if threshold_path is None:
            threshold_path = f"models/threshold{version_suffix}.json"
        if scaler_path is None:
            scaler_path = f"models/scaler{version_suffix}.pkl"
        
        # 加载数据
        data, scaler, original_data = load_data(data_file)
        input_dim = data.shape[1]
        
        # 训练模型
        model, train_losses = train_model_func(
            data,
            input_dim,
            encoding_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # 计算阈值
        threshold, reconstruction_errors = calculate_threshold(model, data, percentile=percentile)
        
        # 保存模型和阈值
        save_model_and_threshold(
            model,
            threshold,
            scaler,
            model_path,
            threshold_path,
            scaler_path,
            config_path
        )
        
        return {
            "status": "success",
            "input_dim": input_dim,
            "total_samples": len(data),
            "epochs": epochs,
            "final_loss": train_losses[-1] if train_losses else None,
            "threshold": float(threshold),
            "percentile": percentile,
            "version": version or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_path": model_path,
            "threshold_path": threshold_path,
            "scaler_path": scaler_path,
            "config_path": config_path or (model_path.replace('.pth', '_config.json'))
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # 运行 MCP 服务器，使用 SSE 传输，端口 3001
    mcp.run(transport="sse", port=3001)
