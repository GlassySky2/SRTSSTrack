import numpy as np


class ReasoningEngine:
    """
    实现 1.3 节中的 ReasoningTrack 思维链逻辑
    根据目标行为 (直线/盘旋) 和场景拓扑 (障碍物) 预测轨迹
    """

    def __init__(self, scene_config=None):
        # 预设场景约束，例如：农田中的建筑物或作物林位置
        self.scene_constraints = scene_config or {"building": [300, 300, 500, 500]}

    def perform_cot_inference(self, last_state, delta_t=1.0):
        """
        核心：思维链推理流程
        last_state: [x, y, vx, vy, ax, ay]
        """
        x, y, vx, vy, ax, ay = last_state

        # Step 1: 运动意图识别 (Action Intent)
        # 逻辑：分析角速度或加速度趋势
        is_maneuvering = np.sqrt(ax ** 2 + ay ** 2) > 2.0

        # Step 2: 时空上下文推断 (ST-Reasoning)
        if is_maneuvering:
            reasoning = "检测到非线性加速度，判定目标正在执行机动避障或盘旋。"
            # 基于圆周运动预测位置
            angle = np.arctan2(vy, vx) + 0.1  # 模拟转向
            speed = np.sqrt(vx ** 2 + vy ** 2)
            pred_x = x + speed * np.cos(angle) * delta_t
            pred_y = y + speed * np.sin(angle) * delta_t
        else:
            reasoning = "目标运动矢量平稳，判定执行 Z 字型农田作业航线。"
            # 基于线性惯性预测位置
            pred_x = x + vx * delta_t + 0.5 * ax * (delta_t ** 2)
            pred_y = y + vy * delta_t + 0.5 * ay * (delta_t ** 2)

        return (pred_x, pred_y), reasoning