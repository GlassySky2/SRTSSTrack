import cv2
import numpy as np
import time
import torch
from pathlib import Path
from ultralytics import YOLOWorld  # 确保安装了 ultralytics

# 导入 SUTrack 核心
from lib.test.tracker.sutrack import SUTRACK
from lib.test.parameter.sutrack import parameters


class YOLO_Semantic_Initializer:
    """感知增强初始化：利用 YOLO-World 的开放词汇能力"""

    def __init__(self, model_path='yolov8s-worldv2.pt'):
        # 第一次运行会下载权重
        self.model = YOLOWorld(model_path)
        # 定义我们要找的语义目标
        self.model.set_classes(["drone", "small aircraft", "uav"])

    def find_target(self, frame, conf=0.15):
        results = self.model.predict(frame, conf=conf, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            # 获取置信度最高的框
            best_box = boxes[0].xyxy[0].cpu().numpy()
            return [float(best_box[0]), float(best_box[1]),
                    float(best_box[2] - best_box[0]), float(best_box[3] - best_box[1])]
        return None


def run_perception_enhanced_system(video_path, save_path):
    # 1. 模块初始化
    params = parameters("sutrack")
    tracker = SUTRACK(params, "drone")
    semantic_eye = YOLO_Semantic_Initializer()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return

    # 2. 初始化：利用 YOLO-World 进行跨模态对齐 (对应论文 1.2 节)
    print("--- 正在进行语义初始化... ---")
    init_box = semantic_eye.find_target(frame)

    if init_box is None:
        print("警告：YOLO-World 未能在首帧发现语义目标，使用默认框")
        init_box = [100, 100, 50, 50]

    tracker.initialize(frame, {"init_bbox": init_box})

    # 3. 实验数据记录
    results = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        # 正常的 SUTrack 追踪
        start_t = time.time()
        out = tracker.track(frame)
        box = out['target_bbox']
        score = out['best_score'].item()

        # 增强逻辑：如果分数低于极值，尝试用 YOLO-World 重新定位（感知重获）
        state = "Stable"
        if score < 0.2:
            state = "Searching"
            re_init_box = semantic_eye.find_target(frame)
            if re_init_box:
                box = re_init_box  # 强制纠偏
                state = "Re-Detected"

        latency = time.time() - start_t
        results.append({
            "Frame": frame_id,
            "Score": round(score, 4),
            "State": state,
            "Latency": round(latency, 4)
        })

        # 可视化 (用蓝色表示感知增强版)
        cv2.rectangle(frame, (int(box[0]), int(box[1])),
                      (int(box[0] + box[2]), int(box[1] + box[3])), (255, 0, 0), 2)
        cv2.putText(frame, f"Perception-Enhanced | {state}", (20, 40), 1, 1.5, (255, 0, 0), 2)

    # 保存实验报表
    import pandas as pd
    pd.DataFrame(results).to_csv(save_path.replace(".mp4", "_perception_results.csv"), index=False)
    cap.release()