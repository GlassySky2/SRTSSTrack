import cv2
import os


def extract_frame_by_number(video_path, frame_number, save_path):
    """
    video_path: 视频文件路径
    frame_number: 想要截取的帧索引（从 0 开始）
    save_path: 保存图片的完整路径
    """
    cap = cv2.VideoCapture(video_path)

    # 获取视频总帧数，防止越界
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"[Error] 帧号 {frame_number} 超过总帧数 {total_frames}")
        return

    # 定位到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    if success:
        # 确保保存的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        print(f"[Success] 第 {frame_number} 帧已保存至: {save_path}")
    else:
        print(f"[Error] 无法读取第 {frame_number} 帧")

    cap.release()


if __name__ == "__main__":
    # 替换为你实际的视频路径
    video_file = "your_drone_video.mp4"

    # 比如你想截取第 100 帧作为测试
    target_frame = 100
    output_image = "perception_module/test_sample.jpg"

    extract_frame_by_number(video_file, target_frame, output_image)