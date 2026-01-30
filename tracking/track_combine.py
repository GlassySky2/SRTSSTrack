import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import time
from scipy.spatial.distance import cdist
from lib.test.tracker.sutrack import SUTRACK
from lib.models.sutrack import build_sutrack
from lib.test.parameter.sutrack import parameters
import threading
import os
import sys
import torch

# 1. 获取 demo.py 所在的路径 (sutrack/test/)
current_file_path = os.path.abspath(__file__)
test_dir = os.path.dirname(current_file_path)

# 2. 定位到父目录 (sutrack/)
root_dir = os.path.dirname(test_dir)

# 3. 关键：将工作目录切换到根目录
os.chdir(root_dir)

# 4. 关键：将根目录加入系统路径，确保能 import lib 文件夹
sys.path.append(root_dir)

print(f"当前工作目录已切换至: {os.getcwd()}")

# 航迹update修改：dist增加的逻辑修改

# --- 全局配置 ---
VIDEO_PATH = './DJI_20260110144757_0010_W_750_1654.mp4'
# VIDEO_PATH = r'D:\datasets\2026.1.15_2\2026.1.15_2\DJI_20260115174111_0002_W.MP4'
SAVE_PATH = "./results/test1.mp4"
# predict() 函数不仅是计算一个预测值，它还会更新滤波器内部的状态变量（即 $\hat{x}_{k|k-1}$）。
# 调优参数
DETECTION_THRESH = 18  # 降低门限以捕捉远处小目标

# WING_MASK_RATIO = (0.28, 0.35)

LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# 追踪器核心参数
WINDOW_SIZE = 10  # 统计窗口大小 (N)
HIT_THRESHOLD = 8  # 确认门限 (M)
MAX_MISSES = 10  #
MATCH_RADIUS = 70  # 关联半径

def get_sky_line(frame):
    h, w = frame.shape[:2]

    # --- 阶段 1: 基础语义分割 (向量化重构) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    _, mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sky_line_base = np.full(w, int(h * 0.3), dtype=np.int32)

    if num_labels > 1:
        # 向量化筛选符合条件的连通域索引
        # 条件：y < h*0.15 且 面积 > h*w*0.05
        areas = stats[1:, cv2.CC_STAT_AREA]
        tops = stats[1:, cv2.CC_STAT_TOP]
        valid_indices = np.where((tops < h * 0.15) & (areas > h * w * 0.05))[0]

        if len(valid_indices) > 0:
            # 找到其中面积最大的索引
            best_idx = valid_indices[np.argmax(areas[valid_indices])] + 1
            full_sky_mask = (labels == best_idx).astype(np.uint8) * 255

            # 向量化提取初始边界：找到每一列从底部向上第一个非零像素的位置
            # np.argmax 返回第一个 True 的索引，通过翻转数组来找“最后一行”
            first_white_from_bottom = np.argmax(full_sky_mask[::-1, :], axis=0)
            # 处理全黑列（如果没有白色，argmax 返回 0）
            has_white = np.any(full_sky_mask, axis=0)
            sky_line_base = np.where(has_white, h - first_white_from_bottom, int(h * 0.3))

    # 向量化单调性约束：使用 np.minimum.accumulate 替代 for 循环
    # 逻辑：每一处的值不能大于前一处的值
    sky_line_base = np.minimum.accumulate(sky_line_base)  # 从左往右
    sky_line_base = np.minimum.accumulate(sky_line_base[::-1])[::-1]  # 从右往左

    # --- 阶段 2: 边缘避障调整 ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 向量化生成 sky_only_mask：利用广播机制生成 [0, sky_line_base] 的掩码
    y_indices = np.arange(h).reshape(-1, 1)
    sky_only_mask = (y_indices < sky_line_base).astype(np.uint8) * 255

    sky_only = cv2.bitwise_and(gray, gray, mask=sky_only_mask)
    edges = cv2.Canny(cv2.GaussianBlur(sky_only, (3, 3), 0), 10, 50)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    edges_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    # 向量化边缘扫描逻辑
    # 获取每一列所有边缘点的 y 坐标，没有边缘的地方设为一个极大值 h
    edge_y_coords = np.where(edges_connected > 0, y_indices, h)
    highest_edge_y = np.min(edge_y_coords, axis=0)

    # 获取每一列边缘的最底部 y 坐标，用于判定是否“触地”
    edge_y_coords_for_bottom = np.where(edges_connected > 0, y_indices, -1)
    lowest_edge_y = np.max(edge_y_coords_for_bottom, axis=0)

    # 计算密度
    edge_counts = np.sum(edges_connected > 0, axis=0)
    span = (lowest_edge_y - highest_edge_y + 1).astype(np.float32)
    density = np.divide(edge_counts, span, out=np.zeros_like(span), where=span > 0)

    # 避障判定条件
    is_touching_ground = np.abs(lowest_edge_y - (sky_line_base - 2)) <= 5
    obstacle_mask = (highest_edge_y < h) & is_touching_ground & (density > 0.4)

    # 更新天际线
    new_sky_line = np.where(obstacle_mask, np.maximum(0, highest_edge_y - 3), sky_line_base)
    # --- 阶段 3: 全局中值滤波 ---
    # 确保数据类型为 uint8，OpenCV 的 medianBlur 要求 src.depth() == 0 (CV_8U)
    # 将 sky_line 限制在 0-255 范围内（如果图像高度超过 255，我们需要特殊处理）

    # 如果你的视频高度 H > 255，建议直接用 numpy 的 median 或换一种平滑方式
    if h <= 255:
        sky_line_img = new_sky_line.reshape(1, -1).astype(np.uint8)
        final_sky_line = cv2.medianBlur(sky_line_img, 9).flatten().astype(np.int32)
    else:
        # 如果高度超过 255，uint8 会溢出，建议改用 cv2.blur (均值滤波)
        # 或者使用信号处理库的 medfilt，或者简单的 numpy 处理
        sky_line_img = new_sky_line.reshape(1, -1).astype(np.float32)
        # blur 函数支持 float32
        final_sky_line = cv2.blur(sky_line_img, (9, 1)).flatten().astype(np.int32)

    return final_sky_line

def cluster_scored_detections(candidates, cluster_dist=60):
    if not candidates:
        return []

    # 1. 预处理：按得分降序排列并转为 Numpy 数组以加速计算
    candidates.sort(key=lambda x: x['score'], reverse=True)

    n = len(candidates)
    scores = np.array([c['score'] for c in candidates])
    boxes = np.array([c['box'] for c in candidates])  # [N, 4] -> x, y, w, h

    # 计算所有目标的中心点 [N, 2]
    centers = np.column_stack([
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2
    ])

    used = np.zeros(n, dtype=bool)
    final_detections = []

    for i in range(n):
        if used[i]:
            continue

        used[i] = True
        current_box = boxes[i].copy()
        current_score = scores[i]
        core_score = scores[i]

        # 初始矩形范围：x1, y1, x2, y2
        x1, y1, x2, y2 = current_box[0], current_box[1], \
            current_box[0] + current_box[2], current_box[1] + current_box[3]

        # 2. 向量化寻找匹配点
        # 仅对未使用的点进行计算
        remaining_indices = np.where(~used)[0]
        if len(remaining_indices) > 0:
            # 动态半径计算
            dynamic_dist = max(cluster_dist, max(current_box[2], current_box[3]) * 1.2)

            # 批量计算当前核心与剩余所有点的距离
            dists = np.linalg.norm(centers[remaining_indices] - centers[i], axis=1)

            # 找出在半径内的点
            match_mask = dists < dynamic_dist
            match_indices = remaining_indices[match_mask]

            if len(match_indices) > 0:
                # 更新边界
                sub_boxes = boxes[match_indices]
                x1 = min(x1, np.min(sub_boxes[:, 0]))
                y1 = min(y1, np.min(sub_boxes[:, 1]))
                x2 = max(x2, np.max(sub_boxes[:, 0] + sub_boxes[:, 2]))
                y2 = max(y2, np.max(sub_boxes[:, 1] + sub_boxes[:, 3]))

                # 分数累加（向量化计算 bonus）
                total_bonus = np.sum(scores[match_indices]) * 0.1
                current_score = min(core_score * 1.3, current_score + total_bonus)

                used[match_indices] = True

        final_detections.append({
            'box': [x1, y1, x2 - x1, y2 - y1],
            'score': current_score
        })

    return final_detections

# --- 航迹管理系统 ---
class Track:
    def __init__(self, box, manager=None, frame_rgb=None):
        self.track_id = -1
        self.box = box  # [x, y, w, h]
        self.misses = 0
        self.is_confirmed = False
        self.st_engine = None  # SUTrack 引擎

        # 验证窗口：N帧中出现M帧则转正
        self.hit_window = [True]

        # 轨迹记录
        self.trace = []

        # 传统匹配用的卡尔曼滤波 (仅用于预备阶段)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05

        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.velocity = np.array([0, 0], dtype=np.float32)
        self.cached_predict = (cx, cy)

    def _activate_sutrack(self, frame_rgb, manager):
        """核心：传统坐标传给深度学习，完成接管"""
        try:
            self.st_engine = SUTRACK(manager.sutrack_params, "UAV",shared_network=manager.shared_net)
            # 用当前传统 CV 找到的最后位置初始化
            self.st_engine.initialize(frame_rgb, {'init_bbox': list(self.box)})
            self.track_id = manager.next_id
            manager.next_id += 1
            self.is_confirmed = True
            print(f"--- [ID:{self.track_id}] 目标被 SUTrack 锁定接管 ---")
        except Exception as e:
            print(f"SUTrack 激活失败: {e}")

    def update_by_cv(self, box, manager, frame_rgb):
        """阶段 A：由传统检测坐标更新"""
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        prev_cx, prev_cy = self.kf.statePost[0][0], self.kf.statePost[1][0]
        self.velocity = np.array([cx - prev_cx, cy - prev_cy])

        self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]], np.float32))
        self.box = box
        self.misses = 0
        self.hit_window.append(True)
        if len(self.hit_window) > WINDOW_SIZE: self.hit_window.pop(0)
        # 3. 转正判定
        if not self.is_confirmed and sum(self.hit_window) >= HIT_THRESHOLD:
            # --- 核心修改：检查当前已激活的 SUTrack 数量 ---
            confirmed_count = sum(1 for t in manager.tracks if t.is_confirmed)

            # 假设我们在 manager 中设置了 max_confirmed 参数，例如 3
            if confirmed_count < manager.max_confirmed:
                print(f"[System] 激活新目标: ID {manager.next_id}")
                self._activate_sutrack(frame_rgb, manager)
            else:
                # 如果名额已满，该目标保持为“待定(Pending)”状态
                # 它依然会跟随检测点移动，但不会消耗 GPU 资源
                if getattr(manager, 'debug', False):
                    print("[System] SUTrack 名额已满，目标保持待定状态")
    def get_predict(self):
        return self.cached_predict

    def update_by_sutrack(self, frame_rgb):
        """阶段 B：由深度学习特征更新 (不再需要外部检测坐标)"""
        # 1. 执行 SUTrack 推理
        out = self.st_engine.track(frame_rgb)

        new_box = out['target_bbox']
        score = out['best_score'].item()

        if score >= 0.6:
            # --- 情况 A: 视觉锁定模式 (High Confidence) ---
            self.misses = 0
            self.box = new_box
            # 提取当前视觉中心
            cx, cy = self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2
            # 【核心修改 1】校准卡尔曼滤波，更新内部速度和状态，防止预测漂移
            self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]], np.float32))
            # 【核心修改 2】更新缓存预测值，用于下一帧 run_system 的空间奖励 (Spatial Bonus)
            self.cached_predict = (cx, cy)
        else:
            # --- 情况 B: 盲跑模式 (Low Confidence / 丢失) ---
            self.misses += 1

            # 【核心修改 3】利用上一帧预测的惯性坐标来更新 box
            # 即使视觉跟丢了，框也会根据卡尔曼滤波预测的轨迹继续移动，而不是停在原地
            pred_cx, pred_cy = self.cached_predict
            w, h = self.box[2], self.box[3]
            self.box = [pred_cx - w / 2, pred_cy - h / 2, w, h]
            # 注意：此时不执行 kf.correct，让卡尔曼保持纯物理预测状态

        # 2. 更新轨迹记录 (无论是否丢失，都记录当前框的中心)
        curr_cx, curr_cy = self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2
        self.trace.append((int(curr_cx), int(curr_cy)))
        if len(self.trace) > 40:
            self.trace.pop(0)

    def mark_missed(self):
        self.misses += 1
        self.hit_window.append(False)
        if len(self.hit_window) > 10: self.hit_window.pop(0)


class TrackManager:
    def __init__(self,sutrack_params=None,max_confirmed = 10):
        self.tracks = []
        self.next_id = 1
        self.sutrack_params = sutrack_params
        self.match_radius = 60  # 匹配像素半径
        # --- 新增：限制最大 SUTrack 数量 ---
        self.max_confirmed = max_confirmed
        print("--- [System] 全局预加载 SUTrack 网络权重 ---")
        self.shared_net = build_sutrack(sutrack_params.cfg)
        self.shared_net.load_state_dict(torch.load(sutrack_params.checkpoint)['net'])
        self.shared_net.cuda().eval()

    def _worker_deep_track(self, confirmed_tracks, frame_rgb):
        """GPU 密集型：SUTrack 自主更新"""
        for t in confirmed_tracks:
            t.update_by_sutrack(frame_rgb)

    def _worker_cv_discovery(self, pending_tracks, detections, frame_rgb):
        """CPU 密集型：从检测点中寻找/验证新目标"""
        # A. 状态预测
        for t in pending_tracks:
            pred = t.kf.predict()
            t.cached_predict = (pred[0][0], pred[1][0])
        # B. 如果没有预备航迹，尝试直接从检测点创建
        if not pending_tracks:
            for det in detections:
                if det.get('score', 0) > 4.5:
                    self.tracks.append(Track(det['box']))
            return

        if not detections:
            for t in pending_tracks: t.mark_missed()
            return

        # C. 向量化匹配 (匈牙利算法)
        num_t = len(pending_tracks)
        num_d = len(detections)
        t_preds = np.array([t.cached_predict for t in pending_tracks]).reshape(num_t, 2)
        d_boxes = np.array([d['box'] for d in detections]).reshape(num_d, 4)
        d_centers = d_boxes[:, :2] + d_boxes[:, 2:] / 2

        cost_matrix = cdist(t_preds, d_centers, metric='euclidean')

        # 应用匹配阈值并执行分配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_t, matched_d = set(), set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.match_radius:
                pending_tracks[r].update_by_cv(detections[c]['box'], self, frame_rgb)
                matched_t.add(r)
                matched_d.add(c)

        # D. 未匹配处理
        for i in range(num_t):
            if i not in matched_t:
                pending_tracks[i].mark_missed()

        for j in range(num_d):
            if j not in matched_d:
                det = detections[j]
                if det.get('score', 0) > 4.5:
                    d_box = det['box']
                    d_cx, d_cy = d_centers[j]

                    is_redundant = False
                    for t in self.tracks:
                        # 获取现有目标的中心
                        t_cx = t.box[0] + t.box[2] / 2
                        t_cy = t.box[1] + t.box[3] / 2
                        dist = np.hypot(d_cx - t_cx, d_cy - t_cy)

                        # --- 核心修改逻辑 ---
                        if t.is_confirmed:
                            # 1. 对于 SUTrack 已锁定的目标，排斥半径要大
                            # 建议设为目标最大边的 1.2 倍，至少 40 像素
                            keep_out_zone = max(max(t.box[2], t.box[3]) * 1.2, 40)
                            if dist < keep_out_zone:
                                is_redundant = True
                                break
                        else:
                            # 2. 对于还在考察期的目标，排斥半径稍小
                            # 防止在同一个位置同时开启两个考察期航迹
                            if dist < 20:
                                is_redundant = True
                                break

                    # 只有不是冗余的点，才创建新航迹
                    if not is_redundant:
                        # 传入 self 是为了让 Track 内部能访问 manager 的参数
                        self.tracks.append(Track(d_box))

    def update(self, detections, frame_rgb):
        # 1. 任务分流
        confirmed = [t for t in self.tracks if t.st_engine is not None]
        pending = [t for t in self.tracks if t.st_engine is None]

        # 2. 多线程并行执行
        t1 = threading.Thread(target=self._worker_deep_track, args=(confirmed, frame_rgb))
        t2 = threading.Thread(target=self._worker_cv_discovery, args=(pending, detections, frame_rgb))

        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # 3. 清理长期失踪的目标 (MAX_MISSES=15)
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]

def detect_on_crop(crop_frame, sky_mask, offset_x=0, offset_y=0):
    # 1. 预处理 (转灰度)
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

    # 2. 优化：先做 CLAHE，再应用 Mask
    # 理由：让 CLAHE 看到完整的亮度分布，再用 Mask 切掉不需要的部分，边缘更平滑
    enhanced = clahe.apply(gray)

    # 3. 掩码处理
    # 缩短腐蚀迭代次数，1次通常足够，节省时间
    refined_mask = cv2.erode(sky_mask, np.ones((3, 3), np.uint8), iterations=1)

    # 将地面区域强行拉高亮度（变成白色），这样黑帽变换（找暗点）就会完全忽略地面
    # 比位运算更安全，因为它消除了地面产生的强边缘
    gray_sky = cv2.bitwise_or(enhanced, cv2.bitwise_not(refined_mask))

    # 4. 黑帽变换 (跳过闭运算，直接提取暗色小目标)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(gray_sky, cv2.MORPH_BLACKHAT, kernel)

    # 5. 二值化与连通域分析
    _, combined = cv2.threshold(blackhat, DETECTION_THRESH, 255, cv2.THRESH_BINARY)
    num, _, stats, _ = cv2.connectedComponentsWithStats(combined)

    # # ================= 可视化拼接逻辑 =================
    # # 为了方便显示，我们将所有图转为BGR格式以便拼接
    # def to_bgr(img):
    #     return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #
    # # 第一行：原图 -> CLAHE增强 -> 掩码填充
    # row1 = np.hstack((crop_frame, to_bgr(enhanced), to_bgr(gray_sky)))
    # # 第二行：黑帽结果 -> 二值化结果 -> 结果标注(在原图上画圈)
    # res_img = crop_frame.copy()
    # num, _, stats, _ = cv2.connectedComponentsWithStats(combined)
    #
    # # 简单的内部绘制逻辑
    # for i in range(1, num):
    #     x, y, w, h, area = stats[i]
    #     if 2 < area < 500:
    #         cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #
    # row2 = np.hstack((to_bgr(blackhat), to_bgr(combined), res_img))
    #
    # # 纵向拼接并缩放
    # debug_board = np.vstack((row1, row2))
    # display_scale = 0.8  # 如果屏幕放不下可以调小
    # show_res = cv2.resize(debug_board, (0, 0), fx=display_scale, fy=display_scale)
    #
    # cv2.imshow("Debug Pipeline: Crop -> CLAHE -> Masked -> Blackhat -> Binary", show_res)
    # cv2.waitKey(1)  # 必须加这一行，窗口才会刷新更新图像
    # # =================================================
    # 6. 速度优化：通过 Numpy 一次性过滤 stats
    # 避免在 Python 循环里处理成千上万个噪点
    if num > 1:
        # 获取除了背景（index 0）以外的所有数据
        valid_stats = stats[1:]
        ws = valid_stats[:, 2]
        hs = valid_stats[:, 3]
        areas = valid_stats[:, 4]

        # 向量化过滤：一次性计算所有纵横比和面积
        aspect_ratios = ws / hs
        mask = (aspect_ratios > 0.3) & (aspect_ratios < 3.0) & (areas > 2) & (areas < 500)

        # 仅对通过筛选的目标生成字典
        indices = np.where(mask)[0]
        cands = [
            {
                'box': (int(valid_stats[idx, 0] + offset_x),
                        int(valid_stats[idx, 1] + offset_y),
                        int(valid_stats[idx, 2]),
                        int(valid_stats[idx, 3])),
                'area': int(valid_stats[idx, 4])
            }
            for idx in indices
        ]
        return cands

    return []

def draw_debug_info(frame, cand):
    x, y, w, h = cand['box']
    # 背景补偿后的运动量、面积分、空间奖励
    # 注意：这些值需要你在 cand 字典里预先存好
    m_p = cand.get('m_part', 0)
    a_p = cand.get('a_part', 0)
    s_p = cand.get('s_bonus', 0)
    total = cand.get('score', 0)

    color = (0, 255, 0) if total > 2.5 else (0, 165, 255)  # 高分绿色，低分橙色

    # 画框
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # 绘制多行文本
    lines = [f"Mot: {m_p:.1f}", f"Are: {a_p:.1f}", f"Spa: {s_p:.1f}", f"Sum: {total:.2f}"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + w + 5, y + 10 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def run_system(video_path, save_path,sutrack_cfg="sutrack_b224"):

    # --- 1. 初始化深度学习追踪参数 ---
    sut_params = parameters(sutrack_cfg)
    sut_params.debug = 0  # 生产环境关闭 debug 以提速

    # --- 2. 基础初始化 ---
    start_time = time.time()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fw, fh = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 3. 传入参数初始化新的 TrackManager ---
    tracker = TrackManager(sutrack_params=sut_params)

    prev_gray = None
    prev_sky_line_f = None
    frame_count = 0
    pts_ref = None

    cfg = {
        'CLUSTER_DIST': 60,
        'ALPHA': 0.3,
        'W_MOTION': 1.5,
        'W_AREA': 1.2,
        'W_SPATIAL': 1.5,
        'MOTION_FLOOR': 1.5,
        'BUFFER_PIXELS': 0,
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # 1. 天际线平滑
        raw_sky_line = get_sky_line(frame)
        if prev_sky_line_f is None:
            sky_line_f = raw_sky_line.astype(np.float32)
        else:
            sky_line_f = cfg['ALPHA'] * raw_sky_line + (1 - cfg['ALPHA']) * prev_sky_line_f
        prev_sky_line_f = sky_line_f.copy()
        sky_line = sky_line_f.astype(np.int32)

        # 2. 准备遮罩
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 为 SUTrack 准备 RGB 图
        global_mask = np.ones((fh, fw), dtype=np.uint8) * 255
        # 建议：只在天际线以下（地面区域）找背景参考点，这样 M 矩阵更稳
        for c in range(fw):
            global_mask[0:sky_line[c], c] = 0

        row_idx = np.arange(fh).reshape(-1, 1)
        sky_mask = (row_idx < sky_line).astype(np.uint8) * 255

        validated_with_score = []
        m_status = "LOST"

        # 3. 运动补偿与检测
        if prev_gray is not None and prev_gray.shape == curr_gray.shape:
            M = None
            # --- 特征点复用逻辑开始 ---
            # 如果没有点，或者存活的点太少（少于 40 个），重新检测特征点
            if pts_ref is None or len(pts_ref) < 40:
                pts_ref = cv2.goodFeaturesToTrack(prev_gray, 120, 0.01, 10, mask=global_mask)
            if pts_ref is not None:
                # 光流追踪上一帧的点到这一帧
                pts_curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_ref, None, **LK_PARAMS)

                # 筛选追踪成功的点
                good_prev = pts_ref[st == 1]
                good_curr = pts_curr[st == 1]

                if len(good_prev) >= 10:  # 至少 10 个点才计算仿射变换
                    M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                    # 【核心优化】将当前帧成功的点存下来，作为下一帧的参考点
                    # 这样下一帧循环时，就不需要跑 goodFeaturesToTrack 了
                    pts_ref = good_curr.reshape(-1, 1, 2)
                else:
                    pts_ref = None  # 点太少，下一帧强制刷新
            # --- 特征点复用逻辑结束 ---

            if M is not None:
                m_status = "LOCKED"
                # ============ 【物理裁剪修改开始】 ============
                # A. 计算裁剪边界：找到天际线最深的点，增加 50 像素缓冲区
                y_cutoff = int(np.max(sky_line)+2)
                y_cutoff = min(fh, y_cutoff)  # 防止越界

                # B. 实施物理裁剪 (生成新的更小的内存矩阵)
                # 只有这里切了，后面的 cvtColor, CLAHE, Blackhat 才会提速
                roi_frame = frame[0:y_cutoff, :]
                roi_sky_mask = sky_mask[0:y_cutoff, :]
                # C. 调用检测函数：传入裁剪后的图
                # 此时 detect_on_crop 内部处理的像素点大幅减少
                cands = detect_on_crop(roi_frame, roi_sky_mask, offset_x=0, offset_y=0)
                # ============ 【物理裁剪修改结束】 ============

                if len(cands) > 0:
                    # --- 向量化优化开始 ---
                    # 1. 整理所有待检测点的中心坐标 (N, 1, 2)
                    pts_list = []
                    for cand in cands:
                        x, y, w, h = cand['box']
                        pts_list.append([x + w / 2, y + h / 2])
                    p0 = np.array(pts_list, dtype=np.float32).reshape(-1, 1, 2)

                    # 2. 批量计算光流
                    p1, st_obj, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK_PARAMS)

                    # 3. 批量计算背景预期坐标
                    expected_pts = cv2.transform(p0, M)

                    # 4. 获取所有追踪器的预测位置用于计算空间奖励 (提前计算)
                    track_preds = [t.get_predict() for t in tracker.tracks] if tracker.tracks else []

                    # 5. 遍历处理结果
                    for i, cand in enumerate(cands):
                        if st_obj[i] == 1:
                            cx, cy = pts_list[i]
                            # 物理边界快速过滤
                            idx_x = max(0, min(fw - 1, int(cx)))
                            if cy > (sky_line[idx_x] + cfg['BUFFER_PIXELS']): continue
                            # 计算净运动分数 (向量化结果)
                            net_motion = np.linalg.norm(p1[i][0] - expected_pts[i][0])
                            # 计算空间奖励
                            spatial_bonus = 0
                            is_near_track = False
                            if track_preds:
                                # 计算当前点到所有追踪器预测点的距离
                                dists = np.sqrt(np.sum((np.array([cx, cy]) - np.array(track_preds)) ** 2, axis=1))
                                min_dist = np.min(dists)
                                if min_dist < 40:
                                    is_near_track = True
                                    spatial_bonus = cfg['W_SPATIAL'] * max(0, (40 - min_dist) / 8.0)
                            # 准入逻辑
                            if (net_motion > cfg['MOTION_FLOOR'] or is_near_track) and (net_motion < 40.0):
                                eff_motion = max(net_motion, 1.0 if is_near_track else 0)
                                motion_part = cfg['W_MOTION'] * eff_motion
                                area_part = cfg['W_AREA'] * np.log1p(
                                    cand.get('area', cand['box'][2] * cand['box'][3]) + 1)
                                score = motion_part + area_part + spatial_bonus
                                if score > 4.0:
                                    cand['score'] = max(0.01, score)
                                    validated_with_score.append(cand)

                    # --- 向量化优化结束 ---
        print("cand_num" ,len(validated_with_score))
        # 4. 聚类与追踪
        final_detections = cluster_scored_detections(validated_with_score, cluster_dist=cfg['CLUSTER_DIST'])
        # --- 核心修改：调用多线程 update ---
        tracker.update(final_detections, frame_rgb)

        # 绘制天际线
        for c in range(fw - 1):
            cv2.line(frame, (c, sky_line[c]), (c + 1, sky_line[c + 1]), (0, 255, 255), 1)

        active_count = 0
        for t in tracker.tracks:
            # 只有 confirmed 且未丢失过久的目标才绘制绿色框
            if t.is_confirmed:
                x, y, w, h = [int(v) for v in t.box]
                active_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{t.track_id}", (x, y - 10), 1, 1, (0, 255, 0), 2)
                # 绘制 SUTrack 轨迹
                if len(t.trace) > 2:
                    for i in range(1, len(t.trace)):
                        cv2.line(frame, t.trace[i - 1], t.trace[i], (0, 255, 255), 1)
            else:
                # 绘制待确认的黄色候选框
                x, y, w, h = [int(v) for v in t.box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

        draw_info_panel(frame, frame_count, m_status, active_count)
        out.write(frame)
        prev_gray = curr_gray.copy()

        if frame_count % 50 == 0:
            print(f"进度: {frame_count}/{total_frames} 帧...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    duration = time.time() - start_time
    print(f"\n  单片处理完成：耗时 {duration:.2f}秒 | 平均速度 {frame_count / duration:.1f} FPS")

def draw_info_panel(frame, frame_count, m_status, uav_count):
    """绘制左上角半透明信息面板"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    txt_color = (255, 255, 255)
    cv2.putText(frame, f"FRAME: {frame_count}", (25, 40), 0, 0.7, txt_color, 2)
    m_color = (0, 255, 255) if m_status == "LOCKED" else (0, 0, 255)
    cv2.putText(frame, f"M-STATUS: {m_status}", (25, 75), 0, 0.7, m_color, 2)
    cv2.putText(frame, f"ACTIVE UAV: {uav_count}", (25, 110), 0, 0.7, (0, 255, 0), 2)


if __name__ == "__main__":
    run_system(VIDEO_PATH, SAVE_PATH,sutrack_cfg="sutrack_b224")