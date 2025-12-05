"""
MCP (Model Context Protocol) 工具封装类
提供 run_detection 方法供客户端调用
"""
import os
import json
from typing import Dict, List, Optional
from detect import AnomalyDetector


class MCPAnomalyDetectionTool:
    """MCP 异常检测工具类"""
    
    def __init__(
        self,
        model_path: str = "models/model.pth",
        threshold_path: str = "models/threshold.json",
        scaler_path: str = "models/scaler.pkl",
        config_path: Optional[str] = None
    ):
        """
        初始化 MCP 工具
        
        Args:
            model_path: 模型文件路径
            threshold_path: 阈值文件路径
            scaler_path: 标准化器文件路径
            config_path: 模型配置文件路径（可选）
        """
        self.model_path = model_path
        self.threshold_path = threshold_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.detector: Optional[AnomalyDetector] = None
        self._ensure_detector_loaded()
    
    def _ensure_detector_loaded(self):
        """确保检测器已加载"""
        if self.detector is None:
            try:
                self.detector = AnomalyDetector(
                    self.model_path,
                    self.threshold_path,
                    self.scaler_path,
                    self.config_path
                )
            except Exception as e:
                raise RuntimeError(f"无法加载检测器: {str(e)}")
    
    def run_detection(
        self,
        file_path: str,
        return_details: bool = True
    ) -> Dict:
        """
        执行异常检测
        
        Args:
            file_path: 待检测数据文件路径（CSV 或 TXT）
            return_details: 是否返回详细信息（包括每行的检测结果）
            
        Returns:
            Dict: 检测结果，包含：
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
            
            # 确保检测器已加载
            self._ensure_detector_loaded()
            
            # 执行检测
            results, original_df = self.detector.detect_from_file(file_path)
            
            # 统计信息
            total = len(results)
            anomalies = sum(1 for r in results if r['is_anomaly'])
            
            # 构建返回结果
            response = {
                "status": "success",
                "total_rows": total,
                "anomaly_count": anomalies,
                "anomaly_ratio": anomalies / total if total > 0 else 0,
                "threshold": float(self.detector.threshold)
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
    
    def reload_model(
        self,
        model_path: Optional[str] = None,
        threshold_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        重新加载模型
        
        Args:
            model_path: 新的模型文件路径（可选）
            threshold_path: 新的阈值文件路径（可选）
            scaler_path: 新的标准化器文件路径（可选）
            config_path: 新的模型配置文件路径（可选）
        """
        if model_path:
            self.model_path = model_path
        if threshold_path:
            self.threshold_path = threshold_path
        if scaler_path:
            self.scaler_path = scaler_path
        if config_path is not None:
            self.config_path = config_path
        
        self.detector = None
        self._ensure_detector_loaded()
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息，包括模型路径、阈值等
        """
        try:
            self._ensure_detector_loaded()
            return {
                "model_path": self.model_path,
                "threshold_path": self.threshold_path,
                "scaler_path": self.scaler_path,
                "threshold": float(self.detector.threshold),
                "device": str(self.detector.device)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# 便捷函数，供直接调用
def run_detection(
    file_path: str,
    model_path: str = "models/model.pth",
    threshold_path: str = "models/threshold.json",
    scaler_path: str = "models/scaler.pkl",
    config_path: Optional[str] = None,
    return_details: bool = True
) -> Dict:
    """
    便捷函数：执行异常检测
    
    Args:
        file_path: 待检测数据文件路径
        model_path: 模型文件路径
        threshold_path: 阈值文件路径
        scaler_path: 标准化器文件路径
        config_path: 模型配置文件路径（可选）
        return_details: 是否返回详细信息
        
    Returns:
        Dict: 检测结果
    """
    tool = MCPAnomalyDetectionTool(model_path, threshold_path, scaler_path, config_path)
    return tool.run_detection(file_path, return_details)


if __name__ == "__main__":
    """测试代码"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python mcp_tool.py <数据文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("=" * 50)
    print("MCP 异常检测工具测试")
    print("=" * 50)
    
    # 创建工具实例
    tool = MCPAnomalyDetectionTool()
    
    # 获取模型信息
    print("\n模型信息:")
    info = tool.get_model_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    # 执行检测
    print(f"\n执行检测: {file_path}")
    result = tool.run_detection(file_path)
    
    # 显示结果
    print("\n检测结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

