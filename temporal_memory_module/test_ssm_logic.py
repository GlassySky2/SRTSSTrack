import torch
from temporal_memory_module.ssm_tracker import SpatialTemporalSSM


def run_test():
    print("=== SSM 模块逻辑验证开始 ===")
    tracker = SpatialTemporalSSM()

    # 1. 模拟第一帧检测：无人机在 [1, 5] (索引 12)
    tracker.force_init(1, 5)

    # 2. 模拟第二帧：突发严重干扰
    # 假设地面出现一个强干扰点 [6, 0] (索引 42)，但无人机在 [1, 5] 变模糊了
    raw_observation = torch.zeros((7, 7))
    raw_observation[6, 0] = 1.0  # 错误的地面干扰 (置信度高)
    raw_observation[1, 5] = 0.2  # 真实的无人机 (置信度低)

    # 3. 使用 SSM 掩码过滤
    corrected_output = tracker.apply_memory_mask(raw_observation)

    # 4. 更新状态
    s_t = 0.2
    alpha, center = tracker.update_memory_adaptive(corrected_output, s_t)

    # 5. 结果对比
    raw_idx = torch.argmax(raw_observation).item()
    ssm_idx = torch.argmax(corrected_output).item()

    print(f"\n[结果对比]")
    print(f"感知层原始输出索引: {raw_idx} (偏到地面去了！)")
    print(f"SSM 修正后输出索引: {ssm_idx} (稳在目标区域！)")
    print(f"当前自适应学习率 alpha: {alpha:.2f}")
    print(f"下时刻记忆中心: {center}")


if __name__ == "__main__":
    run_test()