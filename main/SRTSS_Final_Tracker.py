import cv2
import numpy as np
import os
import torch
import pandas as pd
from pathlib import Path
from collections import deque
from datetime import datetime

# 导入你原有的模块
from lib.test.tracker.sutrack import SUTRACK
from lib.models.sutrack import build_sutrack
from lib.test.parameter.sutrack import parameters

# 导入新增加的模块（确保你之前创建了这些文件）
from perception_module.yolo_world_wrapper import OpenVocabDetector
from reasoning_module.logic_brain_llm import LLMLogicBrain


class SRTSS_Experiment_Manager:
    """实验数据记录器：专门为论文提供图表数据"""

    def __init__(self, video_name):
        self.video_name = video_name
        self.data = []

    def record_frame(self, frame_id, s_t, logic_state, pos_diff, id_sw):
        self.data.append({
            "Frame": frame_id,
            "Confidence": round(float(s_t), 4),
            "Logic_State": logic_state,  # NORMAL, REASONING, INERTIA
            "Pos_Error_Correction": round(float(pos_diff), 2),
            "ID_Switch": id_sw
        })

    def save_report(self, output_path):
        df = pd.DataFrame(self.data)
        report_file = output_path.replace(".mp4", "_exp_report.csv")
        df.to_csv(report_file, index=False)
        # 自动生成摘要
        summary = f"Video: {self.video_name}\n" \
                  f"Avg Confidence: {df['Confidence'].mean():.2f}\n" \
                  f"LLM Intervention Rate: {(df['Logic_State'] != 'NORMAL').mean() * 100:.2f}%\n"
        with open(report_file.replace(".csv", "_summary.txt"), "w") as f:
            f.write(summary)


# 修改你原有的 Track 类，植入“逻辑脑”
class Track:
    def __init__(self, box, track_id, st_engine):
        self.box = box  # [x, y, w, h]
        self.track_id = track_id
        self.st_engine = st_engine
        self.history_vel = deque(maxlen=10)
        self.is_occluded = False

    def predict_by_ssm(self):
        """模拟 SSM 记忆中枢的隐状态推算"""
        if len(self.history_vel) == 0: return self.box
        avg_v = np.mean(self.history_vel, axis=0)
        new_box = self.box.copy()
        new_box[0] += avg_v[0]
        new_box[1] += avg_v[1]
        return new_box


def run_system(video_path, save_path):
    # 1. 初始化所有模块 (感知/认知/记忆)
    detector = OpenVocabDetector()  # 语义发现眼
    logic_brain = LLMLogicBrain()  # 逻辑脑
    exp_manager = SRTSS_Experiment_Manager(Path(video_path).name)

    cap = cv2.VideoCapture(video_path)
    # ... 初始化视频写入器 (省略部分原有代码) ...

    # 初始化第一帧：使用 YOLO-World (对应论文 1.2 节)
    ret, frame = cap.read()
    init_box = detector.find_target(frame, "drone")  # 语义初始化

    # 构建 SUTrack 引擎
    params = parameters("sutrack_b224")
    tracker_engine = SUTRACK(params, "drone_dataset")
    tracker_engine.initialize(frame, {"init_bbox": init_box})

    curr_track = Track(init_box, 1, tracker_engine)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        # 2. 运动层：SUTrack 快速追踪
        out = curr_track.st_engine.track(frame)
        s_t = out['best_score'].item()
        raw_box = out['target_bbox']

        # 3. 认知层：逻辑判定 (对应论文 1.3 节)
        logic_state = "NORMAL"
        final_box = raw_box

        if s_t < 0.25:  # 阈值触发逻辑脑
            # 这里的推理逻辑可以调用 logic_brain.reason
            decision, msg, state = logic_brain.reason(s_t, raw_box, curr_track.predict_by_ssm())
            logic_state = state
            if state == "REASONING":
                final_box = curr_track.predict_by_ssm()

        # 4. 更新与记录
        diff = np.linalg.norm(np.array(final_box[:2]) - np.array(raw_box[:2]))
        curr_track.box = final_box
        exp_manager.record_frame(frame_id, s_t, logic_state, diff, 0)

        # 5. 可视化绘制
        color = (0, 0, 255) if logic_state == "NORMAL" else (0, 255, 255)
        cv2.rectangle(frame, (int(final_box[0]), int(final_box[1])),
                      (int(final_box[0] + final_box[2]), int(final_box[1] + final_box[3])), color, 2)
        cv2.putText(frame, f"State: {logic_state} | Conf: {s_t:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 写入视频...

    exp_manager.save_report(save_path)
    cap.release()