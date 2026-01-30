import numpy as np
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class UAVDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav_predict_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        # 1. 构建帧列表：从 start_frame 到 end_frame
        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(
            base_path=self.base_path,
            sequence_path=sequence_info['path_in_data'],  # 使用相对路径
            frame=frame_num,
            nz=nz,
            ext=ext) for frame_num in range(start_frame, end_frame + 1)]

        # 2. 读取标注
        anno_path = os.path.join(self.base_path, sequence_info['anno_path'])

        # --- 关键修正：只读取第一行的 4 个坐标 ---
        with open(anno_path, 'r') as f:
            line = f.readline().strip()
            # 处理逗号或空格分隔
            bbox = [float(x) for x in line.replace(',', ' ').split()]
            ground_truth_rect = np.array(bbox).reshape(1, 4)

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect,
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = []
        # 数据集根目录
        data_root = os.path.join(self.base_path, 'data')
        # 标注根目录
        anno_root = os.path.join(self.base_path, 'anno')

        # 检查目录是否存在
        if not os.path.exists(data_root):
            print(f"找不到数据目录: {data_root}")
            return []

        # 自动遍历 data 文件夹下的每一个子文件夹（即每个视频序列）
        # 此时 seq_name 会是 DJI_20260110142853_0001_S 这种
        for seq_name in sorted(os.listdir(data_root)):
            # 过滤某一个视频
            # if seq_name != "DJI_20260110144652_0009_Z":  # 只跑这一个
            #     continue
            seq_dir = os.path.join(data_root, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            # 自动匹配标注文件：DJI_xxx.txt
            anno_file = os.path.join(anno_root, f"{seq_name}.txt")

            if not os.path.exists(anno_file):
                print(f"跳过序列 {seq_name}: 找不到标注文件 {anno_file}")
                continue

            # 读取起始帧（txt 的第二行）
            start_frame = 1
            try:
                with open(anno_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        start_frame = int(lines[1].strip())
            except Exception as e:
                print(f"读取 {seq_name} 的起始帧失败，默认使用第1帧")

            # 计算总帧数
            num_frames = len([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])

            sequence_info_list.append({
                "name": seq_name,
                "path": f"data/{seq_name}",  # 这里保持和磁盘结构一致
                "path_in_data": f"data/{seq_name}",
                "startFrame": start_frame,
                "endFrame": num_frames,
                "nz": 5,
                "ext": "jpg",
                "anno_path": f"anno/{seq_name}.txt",
                "object_class": "drone"
            })

        print(f"成功加载了 {len(sequence_info_list)} 个序列进行预测。")
        return sequence_info_list