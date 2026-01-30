import numpy as np
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVTestDataset(BaseDataset):
    """
    专门用于评估的 UAV 测试集类。
    特点：加载全序列真值，支持 IoU 计算和性能绘图。
    """

    def __init__(self):
        super().__init__()
        # 核心：确保使用的是专门的测试路径变量
        self.base_path = self.env_settings.uav_test_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        """
        根据 sequence_name 构建单个测试序列对象
        """
        # --- 核心修正点：移除 'UAV' 层级，保持与 _get_sequence_list 逻辑一致 ---
        # 1. 标注文件路径：UAVTest/anno/DJI_xxx.txt
        anno_path = os.path.join(self.base_path, 'anno', f'{sequence_name}.txt')

        # 2. 图片文件夹路径：UAVTest/data/DJI_xxx
        frames_path = os.path.join(self.base_path, 'data', sequence_name)

        # 3. 读取标注 (全序列读取用于评估)
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"找不到标注文件: {anno_path}")

        ground_truth_rect = load_text(str(anno_path), delimiter=None, dtype=np.float64)

        # 4. 构建帧列表 (确认图片是 5 位数字 00001.jpg)
        num_frames = ground_truth_rect.shape[0]
        frames_list = ['{}/{:05d}.jpg'.format(frames_path, i) for i in range(1, num_frames + 1)]

        # 5. 设置可见性标签
        target_visible = np.ones(num_frames, dtype=np.bool_)

        # 注意：这里数据集标识建议写成 'uav_test'
        return Sequence(sequence_name, frames_list, 'uav_test', ground_truth_rect.reshape(-1, 4),
                        object_class='drone', target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        """
        扫描 anno 目录获取所有序列名
        """
        anno_root = os.path.join(self.base_path, 'anno')
        if not os.path.exists(anno_root):
            print(f"警告: 找不到标注目录 {anno_root}")
            return []

        # 获取文件名（去掉 .txt）
        seq_list = sorted([f.replace('.txt', '') for f in os.listdir(anno_root) if f.endswith('.txt')])
        print(f"--- 数据集扫描报告 ---")
        print(f"根目录: {self.base_path}")
        print(f"成功加载 {len(seq_list)} 个序列用于评估。")
        return seq_list