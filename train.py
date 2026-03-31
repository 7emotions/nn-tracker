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

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)

        self.cap.release()

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
                (x + bw / 2) / w,  # cx
                (y + bh / 2) / h,  # cy
                bw / w,  # w
                bh / h,  # h
            ]

            data.append((self.frames[i], torch.tensor(bbox)))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, bbox = self.data[idx]

        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std

        return frame, bbox


def train_model(
    video_path,
    num_epochs=50,
    batch_size=8,
    save_path="tracker_model_resnet.pth",
    use_pretrained_backbone=True,
):
    """使用预训练ResNet训练模型"""
    print("Creating dataset...")
    dataset = TrackingDataset(video_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Creating model with pretrained ResNet backbone...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TargetTrackingTransformer(
        use_pretrained_backbone=use_pretrained_backbone
    ).to(device)

    trainer = TrainingUtils(model, device)

    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_bbox_loss = 0
        train_conf_loss = 0

        for images, target_bboxes in train_loader:
            images = images.to(device)
            target_bboxes = target_bboxes.to(device)

            losses = trainer.train_step(images, target_bboxes)
            train_loss += losses["total_loss"]
            train_bbox_loss += losses["bbox_loss"]
            train_conf_loss += losses["conf_loss"]

        val_losses = trainer.validate(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"  Train Loss: {train_loss/len(train_loader):.4f}, "
            f"BBox Loss: {train_bbox_loss/len(train_loader):.4f}, "
            f"Conf Loss: {train_conf_loss/len(train_loader):.4f}"
        )
        print(
            f"  Val Loss: {val_losses['total_loss']:.4f}, "
            f"BBox Loss: {val_losses['bbox_loss']:.4f}, "
            f"Conf Loss: {val_losses['conf_loss']:.4f}"
        )

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")

        trainer.scheduler.step()

    print("Training completed!")


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a tracker model")
    parser.add_argument(
        "--video_path", default="test.mp4", type=str, help="Path to the video file"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--save_path",
        type=str,
        default="tracker_model_resnet.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--use_pretrained_backbone",
        action="store_true",
        help="Whether to use a pretrained ResNet backbone",
    )

    args = parser.parse_args()

    train_model(
        args.video_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        use_pretrained_backbone=args.use_pretrained_backbone,
    )
