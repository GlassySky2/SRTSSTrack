import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class OpenVocabDetector:
    """
    1.2 节实战版：基于 CLIP 的开放词项感知核心
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载真实预训练权重
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text_query(self, query):
        """文本编码 L"""
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def extract_visual_v_and_map(self, image_path):
        """
        视觉编码 V，并模拟提取最后一个 Block 的特征图用于融合
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # 获取全局视觉特征 V
            vision_outputs = self.model.vision_model(**inputs, output_hidden_states=True)
            v_global = self.model.visual_projection(vision_outputs.pooler_output)

            # 提取最后一层隐藏状态作为特征图 (Patch tokens)
            # 对于 ViT-B/32，输出通常是 [1, 50, 768]，其中 49 个是 patch 空间特征
            last_hidden_state = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 49, 768]
            feature_map = last_hidden_state.view(1, 7, 7, 768).permute(0, 3, 1, 2)  # [1, 768, 7, 7]

        return v_global / v_global.norm(dim=-1, keepdim=True), feature_map