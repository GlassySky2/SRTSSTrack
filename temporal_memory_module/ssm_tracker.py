import torch
import numpy as np


class SpatialTemporalSSM:
    """
    SRTSS 框架核心：时序状态空间存储模块 (SSM)
    功能：在视觉感知（YOLO-World）波动时，利用隐状态维持轨迹。
    """

    def __init__(self, map_size=(7, 7)):
        self.map_size = map_size
        self.last_center = None  # 隐状态 h_t：记忆中的重心坐标
        self.spatial_mask = torch.ones(map_size)  # 空间先验 M_t
        self.sigma = 1.2  # 搜索半径（高斯核宽度）

    def force_init(self, y, x):
        """
        第一帧强制初始化：锁定目标初始位置
        """
        self.last_center = (float(y), float(x))
        self.spatial_mask = self._generate_gaussian_mask(self.last_center)
        print(f"[SSM] 状态空间已初始化，锚点坐标: ({y}, {x})")

    def _generate_gaussian_mask(self, center, sigma=None):
        """
        生成空间概率掩码：只有记忆点附近的区域才会被赋予高权重
        """
        if sigma is None: sigma = self.sigma
        y_c, x_c = center

        # 创建 7x7 网格坐标
        y, x = torch.meshgrid(
            torch.arange(self.map_size[0]),
            torch.arange(self.map_size[1]),
            indexing='ij'
        )

        # 计算欧氏距离平方
        dist_sq = (x - x_c) ** 2 + (y - y_c) ** 2
        # 生成高斯分布图
        mask = torch.exp(-dist_sq / (2 * sigma ** 2))

        # 归一化到 [0, 1]
        return mask / (mask.max() + 1e-6)

    def apply_memory_mask(self, raw_heatmap):
        """
        【关键步骤】利用记忆过滤感知：S_corrected = S_raw * M_spatial
        """
        return raw_heatmap * self.spatial_mask.to(raw_heatmap.device)

    def update_memory_adaptive(self, corrected_heatmap, s_t):
        """
        根据感知置信度 s_t 自适应更新隐状态
        s_t 越低（如 0.1），alpha 越高（如 0.9），越倾向于维持记忆。
        """
        # 找到当前修正后的最强响应点
        max_idx = torch.argmax(corrected_heatmap).item()
        curr_y = float(max_idx // self.map_size[1])
        curr_x = float(max_idx % self.map_size[1])

        # 论文创新公式：自适应记忆权重 alpha
        # 即使视觉置信度为 0，记忆也会以 0.95 的权重保持
        alpha = max(0.5, min(0.95, 1.0 - s_t))

        if self.last_center is None:
            self.last_center = (curr_y, curr_x)
        else:
            # 状态演化方程：h_t = alpha * h_{t-1} + (1-alpha) * z_t
            new_y = alpha * self.last_center[0] + (1 - alpha) * curr_y
            new_x = alpha * self.last_center[1] + (1 - alpha) * curr_x
            self.last_center = (new_y, new_x)

        # 生成下一帧的预测区域
        self.spatial_mask = self._generate_gaussian_mask(self.last_center)
        return alpha, self.last_center