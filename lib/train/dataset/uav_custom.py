# 新增
import os
import os.path
import torch
import numpy as np
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class UavCustom(BaseVideoDataset):
    """
    适配标准 UAV 格式的自定义数据集加载器
    结构：
    root/data/序列名/*.jpg
    root/anno/序列名.txt
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None,
                 multi_modal_vision=False):  # 仿照 Lasot 新增参数
        """
        root: 数据集根目录 (从 admin/local.py 读取)
        split: 对应 data_specs/uav_custom_train.txt 中的序列清单
        multi_modal_vision: 如果为 True，则将 3 通道图像拼接为 6 通道
        """
        root = env_settings().uav_dir if root is None else root
        super().__init__('UAV', root, image_loader)

        self.sequence_list = self._build_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        # 仿照 Lasot 保存多模态开关状态
        self.multi_modal_vision = multi_modal_vision

    def _build_sequence_list(self, split=None):
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            file_path = os.path.join(ltr_path, 'data_specs', f'uav_custom_{split}.txt')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到序列清单文件: {file_path}")
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        else:
            data_path = os.path.join(self.root, 'data')
            sequence_list = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        return sequence_list

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, 'data', self.sequence_list[seq_id])

    def _read_bb_anno(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        anno_file = os.path.join(self.root, 'anno', f"{seq_name}.txt")
        try:
            # 保持 sep='\s+' 以解决 Tab 分隔符问题
            gt = pandas.read_csv(anno_file, sep='\s+', header=None, dtype=np.float32, na_filter=False).values
        except Exception as e:
            print(f"读取标注失败: {anno_file}, 错误: {e}")
            gt = np.zeros((1, 4), dtype=np.float32)
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:05d}.jpg'.format(frame_id + 1))

    # 仿照 Lasot 新增 _get_frame 处理函数
    def _get_frame(self, seq_path, frame_id):
        frame = self.image_loader(self._get_frame_path(seq_path, frame_id))
        # 如果开启了多模态视觉，且输入是 3 通道，则通过拼接变为 6 通道 (C, H, W -> 2*C, H, W)
        if self.multi_modal_vision:
            frame = np.concatenate((frame, frame), axis=-1)
        return frame

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        # 修改点：使用 self._get_frame 代替直接加载，处理通道拼接
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': 'drone',
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_name(self):
        return 'uav_custom'

    def get_num_sequences(self):
        return len(self.sequence_list)