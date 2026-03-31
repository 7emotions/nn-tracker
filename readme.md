
# 目标追踪器 (Target Tracker)

基于Transformer和OpenCV的目标追踪系统，支持多种追踪器类型和预训练模型。

## 功能特点

- 支持多种追踪器类型：OpenCV KCF、Transformer + 预训练ResNet、Transformer + 自定义模型
- 交互式目标选择：使用鼠标拖拽框选要追踪的目标
- 实时追踪信息显示：FPS、帧数、置信度等
- 支持视频输出保存
- 支持视频文件和摄像头输入
- 灵活的命令行参数配置

## 安装依赖

```bash
conda env create -f env.yml
conda activate transformer_tracker
```

## 快速开始

### 基本用法

```bash
# 使用默认参数（OpenCV KCF追踪器，视频文件为test.mp4）
python demo.py

# 指定视频文件
python demo.py --video test.mp4

# 使用摄像头
python demo.py --video 0
```

### 选择追踪器类型

```bash
# 使用OpenCV KCF追踪器（默认，快速但精度较低）
python demo.py --tracker kcf

# 使用Transformer追踪器 + 预训练ResNet（较慢但精度较高）
python demo.py --tracker resnet

# 使用Transformer追踪器 + 自定义模型
python demo.py --tracker custom --model tracker_model.pth
```

## 命令行参数

- `--video`: 视频文件路径或摄像头索引（默认: test.mp4）
- `--tracker`: 追踪器类型，可选值：
  - `kcf`: OpenCV KCF追踪器（快速，但精度较低）
  - `resnet`: Transformer追踪器 + 预训练ResNet（较慢，但精度较高）
  - `custom`: Transformer追踪器 + 自定义模型（需要提供模型路径）
- `--model`: 预训练模型路径（仅当--tracker=custom时需要）

## 训练模型

### 使用预训练ResNet训练

```bash
python train.py --video test.mp4 \
--use_pretrained_backbone \
--save_path tracker_model.pth \
--num_epochs 50 \
--batch_size 8  
```

这会使用预训练的ResNet50作为特征提取器进行训练，生成的模型文件为`tracker_model.pth`。


## 交互操作

- **鼠标拖拽**: 框选要追踪的目标
- **空格键**: 开始/暂停追踪
- **R键**: 重置选择
- **Q键或ESC**: 退出程序

## 项目结构

```
track/
├── demo.py                   # 主程序（支持命令行参数）
├── transformer_tracker.py    # 追踪器模型实现
├── train.py                  # 训练脚本
├── README.md                 # 项目说明文档
└── test.mp4                  # 示例视频文件
```

## 模型说明

### OpenCV KCF追踪器
- 优点：速度快，实时性好
- 缺点：精度较低，对遮挡和快速运动敏感
- 适用场景：实时追踪，对精度要求不高的场景

### Transformer + 预训练ResNet追踪器
- 优点：精度高，泛化能力强
- 缺点：速度较慢
- 适用场景：离线处理，对精度要求高的场景

### Transformer + 自定义模型追踪器
- 优点：可以针对特定场景优化
- 缺点：需要训练模型
- 适用场景：有特定训练数据，需要针对特定场景优化的场景

## 常见问题

### 1. 模型加载失败

如果遇到模型加载错误，程序会自动回退到OpenCV追踪器。建议重新训练模型：

```bash
python train.py
```

### 2. 追踪效果不好

可以尝试以下方法：
- 使用`--tracker resnet`选项，使用预训练ResNet
- 训练自己的模型：`python train.py`
- 确保选择的目标区域足够大且特征明显

### 3. 速度太慢

可以尝试以下方法：
- 使用`--tracker kcf`选项，使用OpenCV追踪器
- 降低视频分辨率
- 使用GPU加速

## 性能优化

### 使用GPU加速

确保安装了CUDA版本的PyTorch，程序会自动检测并使用GPU。

### 调整视频分辨率

在代码中修改视频分辨率或使用低分辨率视频可以提高速度。

### 选择合适的追踪器

根据应用场景选择合适的追踪器：
- 实时应用：使用OpenCV KCF追踪器
- 离线处理：使用Transformer追踪器

## 贡献

欢迎提交问题和拉取请求。

## 许可证

MIT License
