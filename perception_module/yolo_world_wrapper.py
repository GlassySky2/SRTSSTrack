import torch
import cv2
import numpy as np
from inference.models.yolo_world.yolo_world import YOLOWorld
import supervision as sv


class OpenVocabDetector:
    """
    1.2 节：基于大模型视觉-语言对齐的感知引擎 (直接类加载修复版)
    """

    def __init__(self, model_id="yolo_world/l"):
        # 直接使用类初始化，手动传入 model_id，彻底避开 get_model 的参数冲突
        # 这种方式对于 0.37.1 版本的 inference 库最为稳定
        self.model = YOLOWorld(model_id=model_id)

        # 预设词汇表：利用预训练大模型的零样本能力
        self.classes = ["black uav"]
        self.model.set_classes(self.classes)

    def extract_visual_v_and_map(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        h, w = image.shape[:2]

        # 推理逻辑：旧版本结果处理
        # 显式指定参数名以增强兼容性
        results = self.model.infer(image=image, confidence=0.05)

        # 旧版本返回的可能是列表或单个结果对象，这里做健壮性处理
        if isinstance(results, list):
            results = results[0]

        detections = sv.Detections.from_inference(results)

        # 构建 7x7 语义响应热力图
        raw_heatmap = torch.zeros((7, 7))
        s_t = 0.0

        if len(detections.xyxy) > 0:
            s_t = float(detections.confidence[0])
            box = detections.xyxy[0]

            # 映射像素坐标到 7x7 格点
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            gx = int(cx / (w / 7))
            gy = int(cy / (h / 7))

            if 0 <= gx < 7 and 0 <= gy < 7:
                raw_heatmap[gy, gx] = 1.0

        return s_t, raw_heatmap, detections