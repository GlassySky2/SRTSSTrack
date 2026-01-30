import os


def count_dataset_stats(project_path="Custom_UAV_Dataset_Balanced/images"):
    # 1. 自动定位绝对路径
    current_file_path = os.path.abspath(__file__)
    if "tracking" in current_file_path:
        project_root = os.path.dirname(os.path.dirname(current_file_path))
    else:
        project_root = os.path.dirname(current_file_path)

    data_dir = os.path.join(project_root, project_path)
    lbl_dir = os.path.join(data_dir, "labels")
    img_dir = os.path.join(data_dir, "images")

    if not os.path.exists(lbl_dir):
        print(f"❌ 错误：找不到标签目录 {lbl_dir}")
        return

    total_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

    pos_count = 0
    neg_count = 0

    # 2. 遍历标签文件判断正负
    for lbl_file in label_files:
        file_path = os.path.join(lbl_dir, lbl_file)

        # 判断标准：文件大小为0 或者 内容只包含空白字符
        if os.path.getsize(file_path) == 0:
            neg_count += 1
        else:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    neg_count += 1
                else:
                    pos_count += 1

    # 3. 打印详细报告
    neg_ratio = (neg_count / len(label_files)) * 100 if label_files else 0

    print("-" * 40)
    print(f"📁 数据集路径: {data_dir}")
    print("-" * 40)
    print(f"🖼️  图片总数:    {total_images}")
    print(f"📄 标签总数:    {len(label_files)}")
    print(f"✅ 正样本(有框): {pos_count}")
    print(f"❌ 负样本(背景): {neg_count}")
    print(f"📈 负样本占比:   {neg_ratio:.2f}%")
    print("-" * 40)

    # 4. 给出评估建议
    if neg_ratio > 15:
        print("⚠️ 提示：负样本比例略高，可能会导致模型训练过于保守。")
    elif neg_ratio < 1:
        print("⚠️ 提示：负样本几乎为零，建议采集一些干扰物以降低误报率。")
    else:
        print("🟢 结论：正负样本比例健康，适合开始训练。")
    print("-" * 40)


if __name__ == "__main__":
    count_dataset_stats()