import os
import sys
import glob
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================================
# 1. 路径自动对齐逻辑
# ==========================================================
# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
# 假设脚本在 SUTrack/tracking/ 目录下，向上退一级到达 SUTrack 根目录
if "tracking" in current_file_path:
    project_root = os.path.dirname(os.path.dirname(current_file_path))
else:
    project_root = os.path.dirname(current_file_path)

# 切换进程工作目录到 SUTrack 根目录
os.chdir(project_root)


def verify_dataset(project_name="Custom_UAV_Training_Data", num_samples=20):
    """
    使用 matplotlib 验证 YOLO 格式的数据集
    :param project_name: 数据集文件夹名称
    :param num_samples: 随机抽取的样本数量 (建议使用 4, 9, 16)
    """
    img_dir = os.path.join(project_root, project_name, "images")
    lbl_dir = os.path.join(project_root, project_name, "labels")

    print(f"🔍 正在检索数据集目录: {img_dir}")

    # 获取所有图片文件
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    if not img_files:
        print("❌ 错误：未发现图片！")
        print(f"请检查路径是否正确: {os.path.abspath(img_dir)}")
        print(f"当前工作目录 (CWD): {os.getcwd()}")
        return

    # 随机抽取样本
    num_to_show = min(num_samples, len(img_files))
    samples = random.sample(img_files, num_to_show)

    # 计算网格行列数
    cols = int(num_to_show ** 0.5)
    if cols == 0: cols = 1
    rows = (num_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    # 确保 axes 是数组格式以便遍历
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    print(f"🧐 正在显示 {num_to_show} 张随机样本...")

    for i, img_path in enumerate(samples):
        # 1. 读取并转换颜色空间 (CV2 BGR -> PLT RGB)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # 2. 获取文件名及对应的标签路径
        file_base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{file_base}.txt")

        ax = axes[i]
        ax.imshow(img_rgb)

        # 3. 读取 YOLO 标签并还原坐标
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    if len(parts) == 5:
                        # YOLO 格式: class_id cx cy bw bh
                        _, cx, cy, bw, bh = map(float, parts)

                        # 换算为像素坐标
                        rect_w = bw * w
                        rect_h = bh * h
                        # 计算矩形左上角起始点
                        rect_x = (cx * w) - (rect_w / 2)
                        rect_y = (cy * h) - (rect_h / 2)

                        # 绘制矩形框 (EdgeColor='r' 为红色)
                        rect = patches.Rectangle(
                            (rect_x, rect_y), rect_w, rect_h,
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)

                        # 绘制中心绿点，用于验证中心偏移
                        ax.plot(cx * w, cy * h, 'go', markersize=3)

        # 截取文件名末尾，防止过长重叠
        ax.set_title(f"...{file_base[-25:]}", fontsize=8)
        ax.axis('off')

    # 隐藏多余的子图格子
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行验证
    verify_dataset()