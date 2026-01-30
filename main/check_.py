import cv2
import os

# 配置路径（请确保文件名与你服务器上的完全一致）
video_path = '/home/xyp/sx/SUTrack/data/UAV-Anti-UAV_Train_000271.mp4'
gt_path = '/home/xyp/sx/SUTrack/data/groundtruth_rect.txt'
output_dir = '/home/xyp/sx/SUTrack/data/annotated_frames'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 读取 Ground Truth 坐标
# 假设格式是: x,y,w,h (通常用空格或逗号隔开)
with open(gt_path, 'r') as f:
    # 过滤掉空行，并将每行转为整数列表
    gt_lines = [line.strip().replace(',', ' ').split() for line in f.readlines() if line.strip()]

# 2. 打开视频并渲染
cap = cv2.VideoCapture(video_path)
count = 0

print("开始渲染标注帧...")
while count < 1000:
    ret, frame = cap.read()
    if not ret or count >= len(gt_lines):
        break

    # 解析当前帧坐标
    try:
        # 转换为整数坐标
        x, y, w, h = map(int, map(float, gt_lines[count]))

        # 在图像上画框 (BGR格式，(0,0,255)是红色，2是线条宽度)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 写上帧号
        cv2.putText(frame, f"Frame: {count}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 保存
        save_path = os.path.join(output_dir, f'annotated_{count:04d}.jpg')
        cv2.imwrite(save_path, frame)
        print(f"帧 {count} 已渲染并保存")
    except Exception as e:
        print(f"第 {count} 帧坐标解析失败: {e}")

    count += 1

cap.release()
print(f"\n检查完成！请查看文件夹: {output_dir}")