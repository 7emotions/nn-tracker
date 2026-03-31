# demo_cmd.py - 使用命令行参数的目标追踪器
import cv2
import numpy as np
import sys
import os
import argparse
from pathlib import Path

from transformer_tracker import TargetTracker


class InteractiveTracker:
    """交互式目标追踪器（优化版）"""

    def __init__(self):
        self.tracker = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.bbox = None
        self.tracking = False
        self.video_path = None
        self.cap = None
        self.out = None
        self.frame_count = 0
        self.fps = 0
        self.total_frames = 0
        self.initial_frame = None

        self.use_pretrained_backbone = False
        self.use_transformer = False
        self.model_path = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于选择目标（仅在未追踪时有效）"""
        if not self.tracking:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selecting = True
                self.start_point = (x, y)
                self.end_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
                self.end_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.selecting = False
                if self.start_point and self.end_point:
                    x1 = min(self.start_point[0], self.end_point[0])
                    y1 = min(self.start_point[1], self.end_point[1])
                    x2 = max(self.start_point[0], self.end_point[0])
                    y2 = max(self.start_point[1], self.end_point[1])
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        self.bbox = (x1, y1, x2 - x1, y2 - y1)
                        print(f"选择目标: {self.bbox}")

    def draw_selection_box(self, frame):
        """在帧上绘制正在拖拽的选择框"""
        if self.selecting and self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def draw_tracking_info(self, frame, bbox, confidence, fps):
        """绘制追踪信息"""
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Conf: {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Frame: {self.frame_count}/{self.total_frames}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        status = "TRACKING" if self.tracking else "IDLE"
        color = (0, 255, 0) if self.tracking else (0, 0, 255)
        cv2.putText(
            frame,
            status,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        instructions = "Space: Start/Pause | R: Reset | Q: Quit"
        cv2.putText(
            frame,
            instructions,
            (10, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        return frame

    def run(self, video_source=0):
        """
        运行交互式追踪程序
        1. 打开视频源，显示第一帧，等待用户框选目标
        2. 按空格键开始追踪
        """
        if isinstance(video_source, str):
            self.video_path = video_source
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                print(f"无法打开视频文件: {video_source}")
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"打开视频文件: {self.video_path}")
            print(f"总帧数: {self.total_frames}")
        else:
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {video_source}")
                return
            self.total_frames = 0
            print(f"打开摄像头 {video_source}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = self.cap.get(cv2.CAP_PROP_FPS)

        window_name = "Transformer Target Tracker"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        save_video = False
        output_path = "tracking_output.mp4"
        if save_video and self.video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))
            print(f"保存输出视频到: {output_path}")

        ret, self.initial_frame = self.cap.read()
        if not ret:
            print("无法读取第一帧")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("\n" + "=" * 50)
        print("请用鼠标拖拽框选要追踪的目标，然后按【空格】开始追踪")
        print("按【R】重置选择，按【Q】退出")
        print("=" * 50 + "\n")

        waiting_selection = True
        while waiting_selection:
            display_frame = self.initial_frame.copy()
            if self.bbox is not None:
                x, y, w, h = self.bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            display_frame = self.draw_selection_box(display_frame)

            cv2.putText(
                display_frame,
                "Drag to select target, then press SPACE to start",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and self.bbox is not None:
                waiting_selection = False  # 确认选择
            elif key == ord("r") or key == ord("R"):
                self.bbox = None  # 重置选择
                print("已重置选择，请重新框选目标")
            elif key == ord("q") or key == 27:
                self.cleanup()
                return

        if self.use_transformer:
            self.tracker = TargetTracker(
                model_path=self.model_path,
                use_pretrained_backbone=self.use_pretrained_backbone,
            )
        else:
            self.tracker = TargetTracker()

        self.tracker.init_tracker(self.initial_frame, self.bbox)
        self.tracking = True
        self.frame_count = 0
        print("开始追踪！")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_time = cv2.getTickCount()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("视频结束或读取失败")
                break

            display_frame = frame.copy()

            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            if time_diff > 0:
                fps = 1.0 / time_diff
            else:
                fps = 0
            prev_time = current_time

            if self.tracking and self.tracker:
                try:
                    bbox_norm, confidence = self.tracker.track(frame)

                    # 转换归一化坐标
                    if bbox_norm is not None:
                        cx, cy, w_norm, h_norm = bbox_norm
                        x = int((cx - w_norm / 2) * width)
                        y = int((cy - h_norm / 2) * height)
                        w = int(w_norm * width)
                        h = int(h_norm * height)

                        # 确保边界框在画面内
                        x = max(0, min(x, width))
                        y = max(0, min(y, height))
                        w = min(w, width - x)
                        h = min(h, height - y)

                        self.bbox = (x, y, w, h)

                        display_frame = self.draw_tracking_info(
                            display_frame, self.bbox, confidence, fps
                        )
                    else:
                        cv2.putText(
                            display_frame,
                            "Tracking lost!",
                            (10, height - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                except Exception as e:
                    print(f"追踪错误: {e}")
                    self.tracking = False
            else:
                display_frame = self.draw_selection_box(display_frame)

                if self.bbox is not None and not self.tracking:
                    x, y, w, h = self.bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        "Selected Target",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    display_frame,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    "Press SPACE to start tracking",
                    (10, display_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            if self.out:
                self.out.write(display_frame)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q或ESC退出
                print("\n退出程序")
                break

            elif key == ord(" "):
                if not self.tracking:
                    self.tracker.init_tracker(frame, self.bbox)
                    self.tracking = True

            elif key == ord("r") or key == ord("R"):
                self.reset_tracking()

            self.frame_count += 1

        self.cleanup()

    def reset_tracking(self):
        """重置追踪"""
        self.tracking = False
        self.bbox = None
        self.start_point = None
        self.end_point = None
        self.frame_count = 0
        print("追踪已重置，请重新选择目标")

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="目标追踪器")
    parser.add_argument(
        "--video",
        type=str,
        default="test.mp4",
        help="视频文件路径或摄像头索引 (默认: test.mp4)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="resnet",
        choices=["kcf", "resnet", "custom"],
        help="追踪器类型: kcf(OpenCV), resnet(预训练ResNet), custom(自定义模型) (默认: kcf)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="预训练模型路径 (仅当--tracker=custom时需要)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = InteractiveTracker()
    video_path = args.video

    if args.tracker == "kcf":
        # 使用OpenCV追踪器
        tracker.use_pretrained_backbone = False
        tracker.use_transformer = False
        print("使用OpenCV KCF追踪器")
    elif args.tracker == "resnet":
        # 使用Transformer追踪器 + 预训练ResNet
        tracker.use_pretrained_backbone = True
        tracker.use_transformer = True
        tracker.model_path = args.model if args.model else "tracker_model_resnet.pth"
        print("使用Transformer追踪器 + 预训练ResNet")
    elif args.tracker == "custom":
        # 使用Transformer追踪器 + 自定义特征提取器
        if args.model and os.path.exists(args.model):
            tracker.use_pretrained_backbone = False
            tracker.use_transformer = True
            tracker.model_path = args.model
            print(f"使用Transformer追踪器 + 自定义模型: {args.model}")
        else:
            print("未找到预训练模型，使用OpenCV追踪器")
            tracker.use_pretrained_backbone = False
            tracker.use_transformer = False

    try:
        tracker.run(video_path)
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        tracker.cleanup()


if __name__ == "__main__":
    main()
