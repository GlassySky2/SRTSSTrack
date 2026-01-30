"""
SRTSS-Track: 认知推理与身份修正模块 (Section 1.3)
该模块负责在感知置信度较低时，触发思维链 (CoT) 推理并进行 ID 校验。
"""

from .reasoning_engine import ReasoningEngine
from .identity_manager import IdentityManager

# 暴露统一的调用接口，方便主程序直接使用
__all__ = ["ReasoningEngine", "IdentityManager", "get_module_version"]

def get_module_version():
    """返回模块版本及对应的论文章节"""
    return "SRTSS-Reasoning-v1.0 (Ref: Section 1.3)"

# 初始化日志（
print(f"[Module Load] {get_module_version()} initialized.")