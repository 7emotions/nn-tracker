import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from collections import deque

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def forward(self, x):
        x = self.pos_encoder(x)
        return self.encoder(x)

class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def forward(self, tgt, memory):
        tgt = self.pos_encoder(tgt)
        return self.decoder(tgt, memory)

class TargetTrackingTransformer(nn.Module):
    """基于Transformer的目标追踪模型"""
    def __init__(self, input_dim=512, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, num_queries=100, dropout=0.1,
                 use_pretrained_backbone=False):
        super().__init__()
        
        # 特征提取
        if use_pretrained_backbone:
            # 使用预训练的ResNet作为特征提取器
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # 256 channels
                resnet.layer2,  # 512 channels
            )
            # 冻结预训练层
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            
            # 添加适配层
            self.feature_adapter = nn.Sequential(
                nn.Conv2d(512, d_model, kernel_size=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(256, d_model, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
            )
        
        # 空间压缩
        self.spatial_compress = nn.AdaptiveAvgPool2d((7, 7))
        
        # Transformer组件
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, 
                                          dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers,
                                          dim_feedforward, dropout)
        
        # 目标查询
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))
        
        # 输出头
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # 边界框坐标 [cx, cy, w, h]
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.num_queries = num_queries
        
    def forward(self, x, target_embedding=None):
        """
        x: 输入图像 [batch_size, 3, H, W]
        target_embedding: 目标嵌入 [batch_size, d_model] (可选)
        """
        batch_size = x.shape[0]
        
        # 提取特征
        features = self.feature_extractor(x)  # [B, d_model, H', W']
        
        # 如果使用预训练模型，应用适配层
        if hasattr(self, 'feature_adapter') and self.feature_adapter is not None:
            features = self.feature_adapter(features)
        
        features = self.spatial_compress(features)  # [B, d_model, 7, 7]
        
        # 重塑为序列格式
        features_seq = features.flatten(2).permute(0, 2, 1)  # [B, 49, d_model]
        
        # 编码器
        memory = self.encoder(features_seq)  # [B, 49, d_model]
        
        # 解码器查询
        queries = self.query_embed.repeat(batch_size, 1, 1)  # [B, num_queries, d_model]
        
        # 如果提供了目标嵌入，将其添加到查询中
        if target_embedding is not None:
            # 只将目标嵌入添加到第一个查询
            queries[:, 0:1, :] = queries[:, 0:1, :] + target_embedding.unsqueeze(1)
        
        # 解码器
        outputs = self.decoder(queries, memory)  # [B, num_queries, d_model]
        
        # 预测边界框和置信度
        bboxes = self.bbox_head(outputs)  # [B, num_queries, 4]
        confidences = self.confidence_head(outputs)  # [B, num_queries, 1]
        
        return bboxes, confidences

class TrajectoryTracker:
    """轨迹追踪器，保持追踪的连续性"""
    def __init__(self, max_trajectory_length=30, iou_threshold=0.5):
        self.max_trajectory_length = max_trajectory_length
        self.iou_threshold = iou_threshold
        self.trajectories = {}  # id -> deque of bboxes
        self.next_id = 0
        
    def update(self, detections, confidences, frame_width, frame_height):
        """
        更新轨迹
        detections: 检测到的边界框 [N, 4]
        confidences: 置信度 [N]
        """
        if len(detections) == 0:
            return {}
        
        # 将边界框归一化坐标转换为绝对坐标
        abs_detections = detections.clone()
        abs_detections[:, 0] *= frame_width  # cx
        abs_detections[:, 1] *= frame_height  # cy
        abs_detections[:, 2] *= frame_width   # w
        abs_detections[:, 3] *= frame_height  # h
        
        # 将中心点坐标转换为左上角坐标用于IoU计算
        det_boxes = torch.zeros_like(abs_detections)
        det_boxes[:, 0] = abs_detections[:, 0] - abs_detections[:, 2] / 2
        det_boxes[:, 1] = abs_detections[:, 1] - abs_detections[:, 3] / 2
        det_boxes[:, 2] = abs_detections[:, 0] + abs_detections[:, 2] / 2
        det_boxes[:, 3] = abs_detections[:, 1] + abs_detections[:, 3] / 2
        
        # 匹配检测与现有轨迹
        matched_ids = set()
        matched_det_indices = set()
        
        for traj_id, traj_boxes in self.trajectories.items():
            if len(traj_boxes) == 0:
                continue
                
            last_box = traj_boxes[-1]
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det_box in enumerate(det_boxes):
                if det_idx in matched_det_indices:
                    continue
                    
                iou = self._compute_iou(last_box, det_box)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                matched_ids.add(traj_id)
                matched_det_indices.add(best_det_idx)
                self.trajectories[traj_id].append(det_boxes[best_det_idx])
                if len(self.trajectories[traj_id]) > self.max_trajectory_length:
                    self.trajectories[traj_id].popleft()
        
        # 为未匹配的检测创建新轨迹
        for det_idx in range(len(det_boxes)):
            if det_idx not in matched_det_indices:
                self.trajectories[self.next_id] = deque([det_boxes[det_idx]], 
                                                        maxlen=self.max_trajectory_length)
                self.next_id += 1
        
        # 返回当前帧的追踪结果
        current_tracks = {}
        for traj_id, traj_boxes in self.trajectories.items():
            if len(traj_boxes) > 0:
                current_tracks[traj_id] = traj_boxes[-1]
        
        return current_tracks
    
    def _compute_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            iou = inter_area / (box1_area + box2_area - inter_area)
        else:
            iou = 0.0
            
        return iou

class TargetTracker:
    """主追踪器类"""
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu', use_pretrained_backbone=False):
        self.device = device
        self.model = TargetTrackingTransformer(use_pretrained_backbone=use_pretrained_backbone).to(device)
        self.model.eval()  # 设置为评估模式
        
        if model_path and os.path.exists(model_path):
            try:
                # 尝试加载模型
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded model from {model_path}")
                self.use_transformer = True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using OpenCV tracker as fallback")
                self.use_transformer = False
                self.opencv_tracker = cv2.TrackerKCF_create()
        else:
            print("Warning: No pre-trained model loaded. Using OpenCV tracker as fallback.")
            self.use_transformer = False
            # 使用OpenCV的KCF追踪器作为备选
            self.opencv_tracker = cv2.TrackerKCF_create()
        
        self.trajectory_tracker = TrajectoryTracker()
        self.target_embedding = None
        
    def init_tracker(self, frame, bbox):
        """
        初始化追踪器，设置目标
        frame: 初始帧图像
        bbox: 目标边界框 [x, y, w, h]
        """
        if self.use_transformer:
            # 提取目标嵌入
            with torch.no_grad():
                x, y, w, h = bbox
                # 裁剪目标区域
                target_region = frame[int(y):int(y+h), int(x):int(x+w)]
                target_tensor = self._preprocess_image(target_region).to(self.device)
                
                # 提取目标特征作为嵌入
                features = self.model.feature_extractor(target_tensor)
                if hasattr(self.model, 'feature_adapter') and self.model.feature_adapter is not None:
                    features = self.model.feature_adapter(features)
                self.target_embedding = features.mean(dim=[2, 3])  # [1, d_model]
            
        else:
            # 使用OpenCV追踪器
            self.opencv_tracker.init(frame, tuple(bbox))

        # 初始化轨迹
        frame_height, frame_width = frame.shape[:2]
        normalized_bbox = self._normalize_bbox(bbox, frame_width, frame_height)
        self.trajectory_tracker.update(
            normalized_bbox.unsqueeze(0), 
            torch.tensor([1.0]), 
            frame_width, 
            frame_height
        )
        
    def track(self, frame):
        """
        追踪当前帧中的目标
        frame: 当前帧图像
        返回: 追踪到的边界框和置信度
        """
        if not self.use_transformer:
            # 使用OpenCV追踪器
            success, bbox = self.opencv_tracker.update(frame)
            if success:
                x, y, w, h = bbox
                frame_height, frame_width = frame.shape[:2]
                bbox_norm = torch.tensor([
                    (x + w/2) / frame_width,   # cx
                    (y + h/2) / frame_height,  # cy
                    w / frame_width,            # w
                    h / frame_height            # h
                ])
                return bbox_norm.numpy(), 1.0
            else:
                return None, 0.0

        if self.target_embedding is None:
            raise ValueError("Tracker not initialized. Call init_tracker first.")
        
        with torch.no_grad():
            # 预处理图像
            frame_tensor = self._preprocess_image(frame).to(self.device)
            
            # 模型推理
            bboxes, confidences = self.model(frame_tensor, self.target_embedding)
            
            # 使用第一个查询的预测结果（第一个查询添加了目标嵌入）
            best_bbox = bboxes[0, 0].cpu()
            best_conf = confidences[0, 0].cpu()
            
            # 更新轨迹
            frame_height, frame_width = frame.shape[:2]
            tracks = self.trajectory_tracker.update(
                best_bbox.unsqueeze(0),
                best_conf.unsqueeze(0),
                frame_width,
                frame_height
            )
            
            # 返回最可能的轨迹
            if tracks:
                traj_id = max(tracks.keys(), key=lambda k: self._get_trajectory_confidence(k))
                bbox = tracks[traj_id]
                # 转换为归一化坐标
                bbox_norm = torch.tensor([
                    (bbox[0] + bbox[2]) / 2 / frame_width,  # cx
                    (bbox[1] + bbox[3]) / 2 / frame_height,  # cy
                    (bbox[2] - bbox[0]) / frame_width,        # w
                    (bbox[3] - bbox[1]) / frame_height        # h
                ])
                result = bbox_norm.numpy(), best_conf.item()
                return result
            
            result = best_bbox.numpy(), best_conf.item()
            return result
    
    def _preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, np.ndarray):
            # 调整大小
            image = cv2.resize(image, (224, 224))
            # 转换为张量并归一化
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            # 标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            return image.unsqueeze(0)
        return image
    
    def _normalize_bbox(self, bbox, width, height):
        """归一化边界框"""
        x, y, w, h = bbox
        return torch.tensor([
            (x + w/2) / width,   # cx
            (y + h/2) / height,  # cy
            w / width,            # w
            h / height            # h
        ])
    
    def _get_trajectory_confidence(self, traj_id):
        """获取轨迹的置信度分数"""
        # 基于轨迹长度和最近检测的置信度
        traj = self.trajectory_tracker.trajectories[traj_id]
        return len(traj) / self.trajectory_tracker.max_trajectory_length

class TrainingUtils:
    """训练工具类"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_step(self, images, target_bboxes):
        """
        单步训练
        images: 输入图像 [B, 3, H, W]
        target_bboxes: 目标边界框 [B, 4]
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        pred_bboxes, pred_confidences = self.model(images)
        
        # 计算损失
        # 边界框损失 (L1损失)
        bbox_loss = F.l1_loss(pred_bboxes[:, 0, :], target_bboxes)
        
        # 置信度损失 (二分类交叉熵)
        target_conf = torch.ones(pred_confidences.shape[0], 1).to(self.device)
        conf_loss = F.binary_cross_entropy(pred_confidences[:, 0, :], target_conf)
        
        # 总损失
        total_loss = bbox_loss + conf_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'bbox_loss': bbox_loss.item(),
            'conf_loss': conf_loss.item()
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_bbox_loss = 0
        total_conf_loss = 0
        
        for images, target_bboxes in val_loader:
            images = images.to(self.device)
            target_bboxes = target_bboxes.to(self.device)
            
            pred_bboxes, pred_confidences = self.model(images)
            
            bbox_loss = F.l1_loss(pred_bboxes[:, 0, :], target_bboxes)
            target_conf = torch.ones(pred_confidences.shape[0], 1).to(self.device)
            conf_loss = F.binary_cross_entropy(pred_confidences[:, 0, :], target_conf)
            
            total_loss += (bbox_loss + conf_loss).item()
            total_bbox_loss += bbox_loss.item()
            total_conf_loss += conf_loss.item()
        
        n_batches = len(val_loader)
        return {
            'total_loss': total_loss / n_batches,
            'bbox_loss': total_bbox_loss / n_batches,
            'conf_loss': total_conf_loss / n_batches
        }

# 示例使用
def demo_tracking(video_path, init_bbox):
    """演示追踪功能"""
    # 初始化追踪器
    tracker = TargetTracker(model_path='tracker_model.pth')
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取视频")
        return
    
    # 初始化追踪器
    tracker.init_tracker(frame, init_bbox)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracking_result.mp4', fourcc, 30.0, 
                          (int(cap.get(3)), int(cap.get(4))))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 追踪目标
        bbox, confidence = tracker.track(frame)
        
        # 绘制边界框
        if bbox is not None:
            height, width = frame.shape[:2]
            cx, cy, w, h = bbox
            x = int((cx - w/2) * width)
            y = int((cy - h/2) * height)
            w = int(w * width)
            h = int(h * height)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Conf: {confidence:.2f}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示帧率
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 写入结果
        out.write(frame)
        
        # 显示
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test.mp4"
    init_bbox = [183, 385, 219, 193]  # [x, y, width, height]
    
    # 运行追踪演示
    demo_tracking(video_path, init_bbox)