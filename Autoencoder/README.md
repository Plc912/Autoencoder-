# Autoencoder深度学习时间序列异常检测工具

作者：庞力铖

邮箱：3522236586@qq.com

基于 PyTorch 自编码器（Autoencoder）的时间序列异常检测工具，支持 CSV 和 TXT 格式数据，提供 FastAPI HTTP 接口、SSE 实时推送和 MCP 工具封装。

## 功能特性

- ✅ 基于自编码器的异常检测算法
- ✅ 支持 CSV 和 TXT 格式数据自动识别
- ✅ FastAPI HTTP 接口服务
- ✅ Server-Sent Events (SSE) 实时推送检测结果
- ✅ MCP 工具类封装，便于集成
- ✅ 模型和阈值可保存、加载
- ✅ 自动数据标准化处理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

使用正常数据训练自编码器模型：

```bash
# 使用默认版本号（时间戳）
python train.py --data normal_data.csv --epochs 100

# 指定版本名称
python train.py --data normal_data.csv --version v1 --epochs 100

# 手动指定所有文件路径（不推荐，会覆盖）
python train.py --data normal_data.csv --model models/model.pth --threshold models/threshold.json --scaler models/scaler.pkl --epochs 100
```

参数说明：

- `--data`: 训练数据文件路径（CSV 或 TXT）
- `--model`: 模型保存路径（默认：models/model.pth）
- `--threshold`: 阈值保存路径（默认：models/threshold.json）
- `--scaler`: 标准化器保存路径（默认：models/scaler.pkl）
- `--config`: 模型配置保存路径（可选，默认自动生成：models/model_config.json）
- `--epochs`: 训练轮数（默认：100）
- `--batch_size`: 批次大小（默认：32）
- `--learning_rate`: 学习率（默认：0.001）
- `--encoding_dim`: 编码层维度（可选，默认自动计算）
- `--percentile`: 阈值计算分位数（默认：95）

训练完成后会在 `models/` 文件夹中生成以下文件：

- `models/model.pth`: 模型权重文件
- `models/model_config.json`: 模型配置文件（包含网络结构信息）
- `models/threshold.json`: 异常检测阈值
- `models/scaler.pkl`: 数据标准化器

### 2. 命令行检测

使用训练好的模型进行异常检测：

```bash
python detect.py --data test_data.csv --model models/model.pth --threshold models/threshold.json --scaler models/scaler.pkl --output results.csv
```

参数说明：

- `--data`: 待检测数据文件路径
- `--model`: 模型文件路径（默认：models/model.pth）
- `--threshold`: 阈值文件路径（默认：models/threshold.json）
- `--scaler`: 标准化器文件路径（默认：models/scaler.pkl）
- `--output`: 结果输出文件路径（可选）

### 3. FastMCP 服务器

启动 MCP 服务器（使用 SSE 传输）：

```bash
python server.py
```

服务器将在 `http://127.0.0.1:3001` 启动，自动提供 `/sse` 端点用于 MCP 客户端连接。

#### MCP 工具

服务器提供以下 MCP 工具：

**1. run_detection - 执行异常检测**

```python
{
    "file_path": "test_data.csv",
    "model_path": "models/model.pth",  # 可选，默认 "models/model.pth"
    "threshold_path": "models/threshold.json",  # 可选，默认 "models/threshold.json"
    "scaler_path": "models/scaler.pkl",  # 可选，默认 "models/scaler.pkl"
    "config_path": null,  # 可选
    "return_details": true  # 可选，默认 true
}
```

返回结果包含：

- `status`: 状态 ("success" 或 "error")
- `total_rows`: 总行数
- `anomaly_count`: 异常行数
- `anomaly_ratio`: 异常比例
- `threshold`: 使用的阈值
- `results`: 详细结果列表（每行包含 index, reconstruction_error, is_anomaly）

**2. get_model_info - 获取模型信息**

```python
{
    "model_path": "models/model.pth",  # 可选，默认 "models/model.pth"
    "threshold_path": "models/threshold.json",  # 可选，默认 "models/threshold.json"
    "scaler_path": "models/scaler.pkl",  # 可选，默认 "models/scaler.pkl"
    "config_path": null  # 可选
}
```

**3. train_model - 训练模型**

```python
{
    "data_file": "training_data.csv",
    "version": "v1",  # 可选，模型版本名称（默认使用时间戳）
    "model_path": null,  # 可选，默认使用版本号自动生成
    "threshold_path": null,  # 可选，默认使用版本号自动生成
    "scaler_path": null,  # 可选，默认使用版本号自动生成
    "config_path": null,  # 可选
    "epochs": 100,  # 可选，默认 100
    "batch_size": 32,  # 可选，默认 32
    "learning_rate": 0.001,  # 可选，默认 0.001
    "encoding_dim": null,  # 可选
    "percentile": 95  # 可选，默认 95
}
```

**版本管理说明：**

- 如果不指定 `version`，系统会自动使用时间戳（如：`20241205_143025`）
- 如果不指定文件路径，系统会根据版本号自动生成路径
- 每次训练都会生成新的版本文件，不会覆盖之前的模型

#### MCP 客户端连接

FastMCP 自动提供标准的 MCP SSE 端点，客户端可以通过以下方式连接：

```
http://127.0.0.1:3001/sse
```

MCP 客户端（如 Claude Desktop、Cursor 等）会自动发现和调用可用的工具。

### 4. MCP 工具类

使用 MCP 工具类进行检测：

```python
from mcp_tool import MCPAnomalyDetectionTool, run_detection

# 方式1：使用工具类
tool = MCPAnomalyDetectionTool(
    model_path="models/model.pth",
    threshold_path="models/threshold.json",
    scaler_path="models/scaler.pkl"
)

result = tool.run_detection("test_data.csv")
print(result)

# 方式2：使用便捷函数
result = run_detection("test_data.csv")
print(result)
```

返回结果格式：

```python
{
    "status": "success",
    "total_rows": 1000,
    "anomaly_count": 25,
    "anomaly_ratio": 0.025,
    "threshold": 0.123456,
    "results": [
        {
            "index": 0,
            "reconstruction_error": 0.05,
            "is_anomaly": False
        },
        # ...
    ]
}
```

## 数据格式要求

- **CSV 格式**：标准逗号分隔值文件
- **TXT 格式**：支持制表符、逗号或空格分隔
- **数据要求**：至少包含一列数值数据，非数值列（如时间戳、ID）会自动忽略

示例数据：

```csv
timestamp,value1,value2,value3
2024-01-01,1.2,3.4,5.6
2024-01-02,1.3,3.5,5.7
...
```

## 算法原理

1. **训练阶段**：

   - 使用正常数据训练自编码器
   - 自编码器学习正常数据的特征表示
   - 计算训练数据的重构误差分布
   - 使用重构误差的分位数（默认95%）作为异常检测阈值
2. **检测阶段**：

   - 将待检测数据输入训练好的自编码器
   - 计算重构误差（原始数据与重构数据的均方误差）
   - 如果重构误差超过阈值，则判定为异常

## 注意事项

1. 训练数据应只包含正常样本，不包含异常样本
2. 数据会自动进行标准化处理，使用训练时的标准化参数
3. 模型文件、阈值文件和标准化器文件需要同时存在才能进行检测
4. 如果数据维度与训练时不一致，检测会失败
5. 模型配置文件（`*_config.json`）会在训练时自动生成，用于加载模型时重建网络结构
6. 注意： 不要手动指定 --model、--threshold、--scaler 参数，让系统自动生成带版本号的路径。现在所有文件都在 models/ 文件夹中，并且都有版本号，不会再出现覆盖问题。
