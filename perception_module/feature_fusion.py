import torch
import torch.nn.functional as F


class FeatureFusionLayer:
    """
    1.2 节：跨模态特征融合层
    实现公式：S = Fusion(V, L)
    """

    def fuse_with_memory(self, raw_heatmap, spatial_mask):
        """
        将感知响应与 SSM 记忆掩码进行点乘融合
        """
        # 实现 S_final = S_raw * M_spatial
        corrected_heatmap = raw_heatmap * spatial_mask.to(raw_heatmap.device)

        # 归一化处理
        if corrected_heatmap.max() > 0:
            corrected_heatmap = corrected_heatmap / (corrected_heatmap.max() + 1e-6)

        return corrected_heatmap
    def align_semantics_real(self, feature_map, text_vec):
        """
        feature_map: [1, 768, 7, 7] (来自 ViT-B/32 的特征图)
        text_vec: [1, 512] (来自 CLIP 的文本编码)
        """
        device = feature_map.device

        l_aligned = F.interpolate(
            text_vec.unsqueeze(0),
            size=768,
            mode='linear',
            align_corners=False
        ).squeeze(0).view(1, 768, 1, 1)

        # 2. 计算余弦相似度 (Cosine Similarity)
        # 将特征图与对齐后的文本向量进行点积

        f_norm = F.normalize(feature_map, p=2, dim=1)
        l_norm = F.normalize(l_aligned, p=2, dim=1)

        heatmap = torch.sum(f_norm * l_norm, dim=1).squeeze(0)  # [7, 7]

        # 3. 归一化输出
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min + 1e-6)
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap