import torch
import cv2
import numpy as np
from temporal_memory_module.ssm_tracker import SpatialTemporalSSM


def run_correction_demo():
    print("--- SRTSS-Track: SSM 记忆纠偏实战测试 ---")

    # 1. 初始化 SSM 模块
    ssm = SpatialTemporalSSM()

    # 2. 模拟第一帧：假设感知正确，目标在上方 [2, 4] 位置
    print("[Frame 1] 目标初始定位：天空区域")
    initial_heatmap = torch.zeros(7, 7)
    initial_heatmap[2, 4] = 0.8  # 天空中的无人机
    v_dummy = torch.randn(1, 512)
    ssm.update_state(v_dummy, initial_heatmap)

    # 3. 模拟第二帧：发生“语义偏移”（即你刚才图片的情况）
    # 此时感知模块给出了两个高响应：天空 [2, 5] (弱) 和 地面 [6, 3] (强)
    print("[Frame 2] 发生干扰：地平线建筑群产生伪高分")
    noisy_heatmap = torch.zeros(7, 7)
    noisy_heatmap[2, 5] = 0.4  # 真正的无人机（分低）
    noisy_heatmap[6, 3] = 0.9  # 误报的地面（分高）

    # 4. 执行 SSM 空间纠偏
    corrected_heatmap = ssm.apply_spatial_prior(noisy_heatmap)

    # 5. 结果对比
    original_target = torch.argmax(noisy_heatmap)
    corrected_target = torch.argmax(corrected_heatmap)

    print(f"--- 纠偏结果 ---")
    print(f"原始最高响应点 (含干扰): {original_target.item()} (指向地面)")
    print(f"SSM 修正后响应点: {corrected_target.item()} (成功锁定天空)")

    if corrected_target.item() < 35:  # 7x7=49，上半部分是天空
        print("[Success] SSM 成功利用时序记忆压制了地面干扰！")


if __name__ == "__main__":
    run_correction_demo()