import os
import shutil
import random


def create_balanced_yolo_dataset(src_path="UAV_Training_Data", target_path="UAV_Dataset_Balanced", split_ratio=0.8,
                                 neg_ratio=0.1):
    """
    src_path: 你的原始数据集路径
    target_path: 新的、平衡后的 YOLO 训练路径
    neg_ratio: 负样本占总数的比例 (0.1 = 10%)
    """
    # 1. 定义并创建 YOLO 目录结构
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(target_path, folder), exist_ok=True)

    img_src = os.path.join(src_path, "images")
    lbl_src = os.path.join(src_path, "labels")

    # 2. 分类原始文件
    all_labels = [f for f in os.listdir(lbl_src) if f.endswith('.txt')]
    pos_labels = []
    neg_labels = []

    for f in all_labels:
        if os.path.getsize(os.path.join(lbl_src, f)) > 0:
            pos_labels.append(f)
        else:
            neg_labels.append(f)

    # 3. 计算并抽取负样本
    # 公式：保留负样本数 = (目标比例 * 正样本数) / (1 - 目标比例)
    num_pos = len(pos_labels)
    keep_neg_count = int((neg_ratio * num_pos) / (1 - neg_ratio))
    selected_neg = random.sample(neg_labels, min(len(neg_labels), keep_neg_count))

    final_list = pos_labels + selected_neg
    random.shuffle(final_list)

    # 4. 划分 Train 和 Val
    split_idx = int(len(final_list) * split_ratio)
    train_set = final_list[:split_idx]
    val_set = final_list[split_idx:]

    def copy_files(file_list, subset):
        for lbl_name in file_list:
            # 复制标签
            shutil.copy(os.path.join(lbl_src, lbl_name), os.path.join(target_path, 'labels', subset, lbl_name))
            # 复制图片
            img_name = lbl_name.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(img_src, img_name)):
                shutil.copy(os.path.join(img_src, img_name), os.path.join(target_path, 'images', subset, img_name))

    print(f"🚀 正在处理数据... 正样本: {num_pos}, 抽取的负样本: {len(selected_neg)}")
    copy_files(train_set, 'train')
    copy_files(val_set, 'val')

    # 5. 自动生成 dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(target_path)}
train: images/train
val: images/val

names:
  0: uav
"""
    with open(os.path.join(target_path, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)

    print(f"✅ 处理完成！平衡后的数据集保存在: {target_path}")
    print(f"📊 最终统计: 训练集 {len(train_set)} 张, 验证集 {len(val_set)} 张")


if __name__ == "__main__":
    # 使用你的绝对路径
    create_balanced_yolo_dataset(
        src_path="/home/xyp/sx/SUTrack/Custom_UAV_Training_Data",
        target_path="/home/xyp/sx/SUTrack/Custom_UAV_Dataset_Balanced"
    )