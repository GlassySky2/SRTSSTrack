import cv2
import numpy as np
import time
import pandas as pd
from pathlib import Path

# 导入 SUTrack 核心
from lib.test.tracker.sutrack import SUTRACK
from lib.test.parameter.sutrack import parameters


class CV_Initializer:

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    def find_target(self, frame):
        fgmask = self.fgbg.apply(frame)
        # 闭运算去除噪声
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:  # 限制无人机大小范围
                if area > max_area:
                    max_area = area
                    best_box = cv2.boundingRect(cnt)
        return best_box  # (x, y, w, h)


def run_baseline_system(video_path, save_path):
    # 1. 引擎初始化
    params = parameters("sutrack")
    tracker = SUTRACK(params, "drone")
    initializer = CV_Initializer()

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    # 2. 第一帧：传统 CV 发现目标
    init_box = initializer.find_target(first_frame)
    if init_box is None:
        init_box = (100, 100, 50, 50)  # 如果没发现，强制给个初始位置模拟失败

    tracker.initialize(first_frame, {"init_bbox": init_box})

    # 3. 实验数据记录
    results = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        start_time = time.time()
        # 纯 SUTrack 跟踪
        out = tracker.track(frame)
        box = out['target_bbox']
        score = out['best_score'].item()
        latency = time.time() - start_time

        # 模拟 Baseline 的缺陷：当分数低时，框会漂移
        state = "Tracking"
        if score < 0.35:
            state = "Lost/Drift"

            # 记录数据
        results.append({
            "Frame": frame_id,
            "Score": round(score, 4),
            "State": state,
            "Latency": round(latency, 4)
        })

        # 可视化
        cv2.rectangle(frame, (int(box[0]), int(box[1])),
                      (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"Baseline | Score: {score:.2f}", (20, 40), 1, 1.5, (0, 255, 0), 2)


    # 导出 CSV 用于对比实验
    pd.DataFrame(results).to_csv(save_path.replace(".mp4", "_baseline_results.csv"), index=False)
    cap.release()


if __name__ == "__main__":
    run_baseline_system("test_samples/testvideos.mp4", "results/testvideos.mp4")