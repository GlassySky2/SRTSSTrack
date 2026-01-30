import torch
from transformers import pipeline


class LLMLogicBrain:
    """
    认知推理模块 (LLM 增强版)
    利用 TinyLLaMA 进行 CoT 思维链推理
    """

    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print("正在加载轻量化逻辑脑模型 (TinyLLaMA)...")
        # 强制使用 CPU 运行，并开启 4-bit 量化 (如果支持) 或半精度以节省内存
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float32,
            device_map="cuda"
        )
        self.template = "<|system|>\nYou are a UAV tracking expert. Analyze the data and answer ONLY 'Reliable' or 'Unreliable'.\n<|user|>\n"

    def reason(self, s_t, vision_pos, ssm_pos):
        # 1. 将数值转换为语义 Prompt
        prompt = (
            f"Current detection: confidence {s_t:.2f} at position {vision_pos}. "
            f"History predicted position {ssm_pos}. "
            f"Is the current detection reliable? Answer with one word."
        )

        full_prompt = f"{self.template}{prompt}\n<|assistant|>\n"

        # 2. 调用 LLM 推理
        outputs = self.pipe(full_prompt, max_new_tokens=10, do_sample=False)
        response = outputs[0]["generated_text"].split("<|assistant|>\n")[-1].strip()

        # 3. 解析逻辑决策
        is_reliable = "Reliable" in response and "Unreliable" not in response

        if is_reliable:
            return vision_pos, f"LLM Decision: {response} (Trust Vision)", "NORMAL"
        else:
            return ssm_pos, f"LLM Decision: {response} (Trust Memory)", "REASONING"