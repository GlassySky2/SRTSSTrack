class IdentityManager:
    """
    实现 1.3 节中的身份修正机制
    利用推理位置作为先验，修正视觉检测的漂移或误报
    """
    def __init__(self, dist_threshold=60):
        self.dist_threshold = dist_threshold

    def correct_identity(self, predicted_pos, detected_boxes):
        """
        在多个候选框中寻找最符合推理逻辑的目标
        """
        if not detected_boxes:
            return None, "No candidates"

        best_idx = -1
        min_dist = float('inf')

        for i, box in enumerate(detected_boxes):
            # 计算欧氏距离
            dist = ((box[0] - predicted_pos[0])**2 + (box[1] - predicted_pos[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        if min_dist < self.dist_threshold:
            return best_idx, f"Success (Dist: {min_dist:.2f})"
        else:
            return None, "Identity Mismatch (Occlusion persistent)"