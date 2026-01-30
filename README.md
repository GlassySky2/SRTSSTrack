### 大模型原理与应用作业

借鉴SUTrack的统一化设计思想，整合多模态先验信息的追踪前端。打破传统追踪算法中“特征-位置”的简单映射，建立一种具备语义理解与逻辑回溯能力的认知追踪体系。
* **统一语义感知 (Perception)**：通过多模态特征编码器将异构输入（RGB图像与自然语言指令）映射至共享语义空间，实现跨模态目标的精准锚定（Grounding）。
*  **认知推理增强 (Reasoning)**：该模块在视觉底层信号发生剧烈波动或完全遮挡时被触发，通过思维链（Chain-of-Thought, CoT）技术补全目标的时空缺失信息。
*  **时序状态空间存储 (Memory)**：相较于传统的卡尔曼滤波，该模块通过非线性状态方程维持目标在长序列中的动态表征，显著提升了在视角突变场景下的身份保持能力。



Please consider citing the original SUTrack paper:

```bibtex
@inproceedings{sutrack,
  title={SUTrack: Towards Simple and Unified Single Object Tracking},
  author={Chen, Xin and Kang, Ben and Geng, Wanting and Zhu, Jiawen and Liu, Yi and Wang, Dong and Lu, Huchuan},
  booktitle=AAAI,
  year={2025}
}

