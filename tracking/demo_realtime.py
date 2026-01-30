import cv2
import torch
import os
import sys

# 1. 环境路径设置 (确保能找到 lib 文件夹)
# 假设你在 /home/xyp/sx/SUTrack/ tracking/ 目录下运行
prj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(prj_root)  # 切换工作目录到根目录，解决权重找不到的问题
sys.path.append(prj_root)

from lib.test.tracker.sutrack import SUTRACK
from lib.test.parameter.sutrack import parameters


def main():
    # 2. 加载参数并补齐缺失属性
    params = parameters("sutrack_b224")
    params.debug = 0  # 必须设为 0，否则远程 SSH 会崩溃

    # 3. 实例化追踪器
    tracker = SUTRACK(params, "UAV")

    # 4. 视频源检查 (请确保 test.mp4 在 SUTrack 根目录下)
    video_input = "DJI_20260110144757_0010_W_750_1654.mp4"
    if not os.path.exists(video_input):
        print(f"错误：找不到输入视频文件 {os.path.abspath(video_input)}")
        return

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print("错误：OpenCV 无法打开视频文件，请检查解码器或路径")
        return

    # 5. 设置视频输出
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_path = "result_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # 6. 初始化第一帧目标
    # 注意：远程无法 selectROI，请先在本地确定目标位置
    # 这里假设一个目标框 [x, y, w, h]
    init_box = [400, 200, 60, 40]

    ret, frame = cap.read()
    if not ret:
        print("无法读取第一帧")
        return

    # SUTrack 需要 RGB 格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracker.initialize(frame_rgb, {'init_bbox': init_box})
    print(f"成功初始化目标: {init_box}")

    # 7. 循环处理
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 推理
        out = tracker.track(frame_rgb)

        # 确保数据回到 CPU 并转为基本类型
        res_box = [int(b) for b in out['target_bbox']]
        # 核心修复点：使用 .item()
        score = out['best_score'].item() if torch.is_tensor(out['best_score']) else out['best_score']

        # 在图像上绘制结果
        x, y, w, h = res_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 这里的 f-string 现在可以正常工作了
        cv2.putText(frame, f"Score: {score:.2f}", (max(0, x), max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 写入视频文件
        out_video.write(frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"已处理 {frame_idx} 帧，置信度: {score:.2f}")

    cap.release()
    out_video.release()
    print(f"处理完成！结果保存在: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()