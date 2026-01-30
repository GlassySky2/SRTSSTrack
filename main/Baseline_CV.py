import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from lib.test.tracker.sutrack import SUTRACK
from lib.test.parameter.sutrack import parameters


def load_groundtruth(txt_path):
    """加载真值文件"""
    return np.loadtxt(txt_path, delimiter=',').reshape(-1, 4)


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
    inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def run_baseline_test(video_path, gt_path):
    # 1. 初始化引擎
    params = parameters("sutrack")
    tracker = SUTRACK(params, "drone")

    cap = cv2.VideoCapture(video_path)
    gt_boxes = load_groundtruth(gt_path)

    ret, frame = cap.read()
    if not ret: return

    # 使用真值的第一行进行初始化 (模拟完美启动)
    init_box = gt_boxes[0]
    tracker.initialize(frame, {"init_bbox": init_box})

    results = []
    print(">>> Baseline 测试启动，正在评估量化指标...")

    frame_idx = 0
    while cap.isOpened() and frame_idx < len(gt_boxes):
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 追踪
        start_t = time.time()
        out = tracker.track(frame)
        pred_box = out['target_bbox']
        score = out['best_score'].item()
        latency = time.time() - start_t

        # 计算指标
        gt_box = gt_boxes[frame_idx]
        iou = calculate_iou(pred_box, gt_box)

        # 记录
        results.append({
            "frame": frame_idx,
            "iou": iou,
            "score": score,
            "latency": latency
        })

        # 可视化对比：红色为真值，绿色为预测
        cv2.rectangle(frame, (int(gt_box[0]), int(gt_box[1])),
                      (int(gt_box[0] + gt_box[2]), int(gt_box[1] + gt_box[3])), (0, 0, 255), 2)
        cv2.rectangle(frame, (int(pred_box[0]), int(pred_box[1])),
                      (int(pred_box[0] + pred_box[2]), int(pred_box[1] + pred_box[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"IoU: {iou:.2f}", (20, 40), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Baseline Evaluation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 保存 CSV
    df = pd.DataFrame(results)
    df.to_csv("baseline_metrics.csv", index=False)

    # 输出汇总数据
    print(f"\n--- Baseline 实验报告 ---")
    print(f"平均 IoU: {df['iou'].mean():.4f}")
    print(f"成功率 (IoU > 0.5): {(df['iou'] > 0.5).mean() * 100:.2f}%")
    print(f"平均 FPS: {1 / df['latency'].mean():.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_baseline_test("data/your_video.mp4", "data/your_video.txt")