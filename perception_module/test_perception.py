import cv2
import torch
import os
from perception_module.yolo_world_wrapper import OpenVocabDetector
import supervision as sv


def test_perception_unit():
    print("--- 单元测试: 感知层能力验证 (兼容模式) ---")
    detector = OpenVocabDetector()

    # 自动定位测试图
    test_img = "perception_module/test_sample.jpg"
    if not os.path.exists(test_img):
        print(f"错误: 找不到 {test_img}，请确认文件位置。")
        return

    s_t, heatmap, detections = detector.extract_visual_v_and_map(test_img)

    print(f"检测置信度 s_t: {s_t:.4f}")
    print(f"7x7 响应矩阵分布:\n{heatmap.numpy()}")

    # 可视化
    image = cv2.imread(test_img)
    # 兼容性写法：使用旧版 supervision 或新版通用接口
    box_annotator = sv.BoxAnnotator()
    annotated_img = box_annotator.annotate(scene=image, detections=detections)

    if len(detections.xyxy) > 0:
        idx = torch.argmax(heatmap).item()
        gy, gx = idx // 7, idx % 7
        cv2.putText(annotated_img, f"SSM Grid: ({gy},{gx})", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite("perception_module/perception_test_result.jpg", annotated_img)
    print("测试成功！可视化结果保存至: perception_test_result.jpg")


if __name__ == "__main__":
    test_perception_unit()