
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from transformer_tracker import TargetTrackingTransformer, TrainingUtils
import os
import torchvision.models as models

class TrackingDataset(Dataset):
    """追踪数据集"""
    def __init__(self, video_path, transform=None):
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transform
        self.frames = []

        # 读取所有帧
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)

        self.cap.release()

        # 生成模拟的追踪数据
        self.data = self._generate_tracking_data()

    def _generate_tracking_data(self):
        """生成模拟的追踪数据"""
        data = []
        for i in range(len(self.frames) - 1):
            # 随机选择一个边界框
            h, w = self.frames[i].shape[:2]
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            bw = np.random.randint(50, 100)
            bh = np.random.randint(50, 100)

            # 添加一些随机运动
            if i > 0:
                dx = np.random.randint(-10, 10)
                dy = np.random.randint(-10, 10)
                x = max(0, min(x + dx, w - bw))
                y = max(0, min(y + dy, h - bh))

            # 归一化边界框
            bbox = [
                (x + bw/2) / w,  # cx
                (y + bh/2) / h,  # cy
                bw / w,          # w
                bh / h           # h
            ]

            data.append((self.frames[i], torch.tensor(bbox)))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, bbox = self.data[idx]

        # 预处理图像
        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0

        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std

        return frame, bbox

def train_model_with_resnet(video_path, num_epochs=50, batch_size=8, save_path='tracker_model_resnet.pth'):
    """使用预训练ResNet训练模型"""
    # 创建数据集
    print("Creating dataset...")
    dataset = TrackingDataset(video_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型，使用预训练ResNet作为特征提取器
    print("Creating model with pretrained ResNet backbone...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TargetTrackingTransformer(use_pretrained_backbone=True).to(device)

    # 创建训练工具
    trainer = TrainingUtils(model, device)

    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        train_bbox_loss = 0
        train_conf_loss = 0

        for images, target_bboxes in train_loader:
            images = images.to(device)
            target_bboxes = target_bboxes.to(device)

            losses = trainer.train_step(images, target_bboxes)
            train_loss += losses['total_loss']
            train_bbox_loss += losses['bbox_loss']
            train_conf_loss += losses['conf_loss']

        # 验证
        val_losses = trainer.validate(val_loader)

        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, "
              f"BBox Loss: {train_bbox_loss/len(train_loader):.4f}, "
              f"Conf Loss: {train_conf_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_losses['total_loss']:.4f}, "
              f"BBox Loss: {val_losses['bbox_loss']:.4f}, "
              f"Conf Loss: {val_losses['conf_loss']:.4f}")

        # 保存最佳模型
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")

        # 更新学习率
        trainer.scheduler.step()

    print("Training completed!")

if __name__ == "__main__":
    # 训练模型
    video_path = 'test.mp4'  # 替换为您的视频文件路径
    train_model_with_resnet(video_path, num_epochs=50, batch_size=8, save_path='tracker_model_resnet.pth')
