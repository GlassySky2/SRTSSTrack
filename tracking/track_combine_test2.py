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
from collections import deque
from ultralytics import YOLO
import torch
# # 如果 echo $DISPLAY 看到的是别的，这里就改成别的
# if 'DISPLAY' not in os.environ:
#     os.environ['DISPLAY'] = 'localhost:10.0'
#
# # 禁用 OpenCV 的某些可能导致崩溃的插件
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
# 创建保存失败样本的目录

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
DEBUG_SAVE_PATH = "debug_yolo_miss"
os.makedirs(DEBUG_SAVE_PATH, exist_ok=True)
# 航迹update修改：dist增加的逻辑修改


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
# MAX_MISSES = 60  #
CONFIRMED_LIMIT = 40  # SUTrack 锁定目标最大允许丢 40 帧
PENDING_LIMIT = 10    # 考察期目标最大允许丢 10 帧
MATCH_RADIUS = 40  # 关联半径
Neg_count = 7 # 被yolo检测八次不是
# --- 航迹管理系统 ---
class Track:
    def __init__(self, box, manager=None, frame_rgb=None):
        self.track_id = -1
        self.box = box  # [x, y, w, h]
        self.misses = 0
        self.is_confirmed = False
        self.st_engine = None  # SUTrack 引擎

        self.is_drone_confirmed = False# 是否通过了 YOLO 验证

        self.negative_count = 0  # 新增：记录 YOLO 连续检测不到目标的次数
        self.yolo_score = 0.0

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
            # 1. 获取当前的原始框 (由传统检测得到，通常很小)
            x, y, w, h = self.box
            cx, cy = x + w / 2, y + h / 2

            # 2. 设置扩边比例或像素
            # 建议：在碎点检测场景下，将框扩充到至少 20-30 像素，或增加 20% 的边缘
            padding_w = max(w * 0.1, 4)  # 至少左右各加 8 像素
            padding_h = max(h * 0.1, 4)  # 至少上下各加 8 像素

            new_w = w + padding_w * 2
            new_h = h + padding_h * 2

            # 3. 边界检查：防止扩大的框超出图像边界
            img_h, img_w = frame_rgb.shape[:2]
            new_x = max(0, cx - new_w / 2)
            new_y = max(0, cy - new_h / 2)
            new_w = min(img_w - new_x, new_w)
            new_h = min(img_h - new_y, new_h)

            # 4. 更新 self.box 为扩大的框
            self.box = [new_x, new_y, new_w, new_h]

            # 5. 初始化引擎 (此时传入的就是“大框”) 起批：SUTrack 初始化 (保持接管速度) ---
            self.st_engine = SUTRACK(manager.sutrack_params, "UAV", shared_network=manager.shared_net)
            self.st_engine.initialize(frame_rgb, {'init_bbox': list(self.box)})
            # 用当前传统 CV 找到的最后位置初始化
            self.track_id = manager.next_id
            manager.next_id += 1
            self.is_confirmed = True
            print(f"--- [ID:{self.track_id}] 目标被 SUTrack 锁定接管 ---")

            #  同步检测：调用 YOLO 进行身份验证
            # 注意：这里调用我们之前对齐训练逻辑的 verify_by_yolo
            is_uav, conf = self.verify_by_yolo(frame_rgb, manager.yolo_model)

            self.is_drone_confirmed = is_uav
            self.yolo_score = conf

            if is_uav:
                print(f"✅ [ID:{self.track_id}] 已起批：YOLO 确认为无人机 (Conf:{conf:.2f})")
            else:
                print(f"⚠️ [ID:{self.track_id}] 已起批：YOLO 尚未发现目标，可能是干扰物")
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

    def update_by_sutrack(self, frame_rgb,yolo_model):
        """阶段 B：由深度学习特征更新 (不再需要外部检测坐标)"""
        """
            逻辑：
            1. Score > 0.35: SUTrack 接管
            2. Score <= 0.35: YOLO 尝试纠偏
            3. 两者皆墨: 惯性导航并标记丢失
        """
        # 1. 执行 SUTrack 推理
        out = self.st_engine.track(frame_rgb)
        # 提取 SUTrack 认为的位置和分数
        sot_box = out['target_bbox']
        sot_score = out['best_score'].item()
        old_box = self.box  # 备份一下，以防万一
        self.box = sot_box  # 先更新为当前帧建议位置
        # 2. 身份复核：无论 SOT 分数高低，每帧都跑一次 YOLO 验证 (或者你也可以设为每隔2帧跑一次以省性能)
        is_uav, yolo_conf = self.verify_by_yolo(frame_rgb, yolo_model)

        # 计算中心点用于卡尔曼和轨迹
        cx, cy = sot_box[0] + sot_box[2] / 2, sot_box[1] + sot_box[3] / 2
        # # 2. 状态记录
        # if score >= 0.35:
        #     self.misses = 0
        #     # 只有在高分时才校准卡尔曼，防止低分噪点把卡尔曼的速度矢量带歪
        #     self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]], np.float32))
        # elif score <= 0.2:
        #     self.mark_missed(step=2)
        # else:
        #     self.mark_missed(step=1)
        # 2. 状态判定与处理逻辑
        if sot_score >= 0.35:# --- 情况 A：SUTrack 表现优秀 ---
            # --- 情况 A：SUTrack 表现优秀 ---
            self.box = sot_box
            if is_uav:
                # 只有 YOLO 也确认是无人机，才认为是真正的“锁定”
                self.misses = 0
                self.is_drone_confirmed = True
                self.negative_count = 0
                # 只有身份确定的高分目标才校准卡尔曼，防止被假目标带歪速度矢量
                self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]], np.float32))
                print(f"--- [ID:{self.track_id}] : 跟踪分高 ({sot_score:.2f}) YOLO认出) ---")
            else:
                # yolo识别不准
                if sot_score >= 0.45 :# neg_count ++ 了已经
                    print(f"--- [ID:{self.track_id}] 警告: 跟踪分高({sot_score:.2f}) 但 YOLO 未认出 (Neg:{self.negative_count}) ---")
                else:
                    self.misses = self.misses + 1
                    print(f"--- [ID:{self.track_id}] 警告: 跟踪分中等({sot_score:.2f}) YOLO 未认出 (Neg:{self.negative_count}) ---")
        elif 0.001 <= sot_score < 0.35:
            # --- 情况 B：SUTrack 虚弱，看 YOLO 能否救回 ---
            print(f"--- [ID:{self.track_id}] 跟踪分低({sot_score:.2f})，尝试 YOLO 验证... ---")
            self.box = sot_box
            # 【核心逻辑修改】：判断 YOLO 分数是否达到强力纠偏阈值 0.6
            if is_uav and yolo_conf >= 0.6:
                # YOLO 极度自信，强制重置 misses 和相关状态
                self.box = sot_box
                self.misses = 0  # <--- 关键：高分 YOLO 不计入丢失
                self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]], np.float32))
                print(f"      >>> [ID:{self.track_id}]  (Conf:{yolo_conf:.2f}) YOLO 强力救回 ")
            elif is_uav and yolo_conf < 0.6:
                # YOLO 虽认出是 UAV 但分数不高，维持航迹但不重置 misses
                self.box = sot_box
                self.mark_missed(step=1)  # <--- 增加 misses 观察
                print(f"      >>> [ID:{self.track_id}]   (Conf:{yolo_conf:.2f}) YOLO 弱验证通过，维持观察...")
            else:
                # --- 情况 C：分数低，彻底跟丢，
                self.box = sot_box
                self.mark_missed(step=2)  # <--- 增加 misses 观察
                print(f"      >>> [ID:{self.track_id}]  (Conf:{yolo_conf:.2f}) YOLO 弱验证未通过,增加misses...")
        else:#跟丢了
            self.box = sot_box
            self.mark_missed(step=3)  # <--- 增加 misses 观察
            print(f"      >>> [ID:{self.track_id}]  跟踪分太低")

        # 4. 记录轨迹与后处理
        self.trace.append((int(cx), int(cy)))
        if len(self.trace) > 20: self.trace.pop(0)

        self.cached_predict = (cx, cy)


    def verify_by_yolo(self, frame_rgb, model):
        """
        对齐训练时的切片逻辑：正方形、32对齐、min_size=128
        """
        fh, fw = frame_rgb.shape[:2]
        tx, ty, tw, th = self.box  # 使用当前的 box
        cx_raw, cy_raw = tx + tw / 2.0, ty + th / 2.0

        # 1. 严格对齐你训练代码 DynamicDatasetCollector 的逻辑
        padding_factor = 2.5
        min_size = 128

        side = max(tw, th) * padding_factor
        crop_size = int(np.ceil(side / 32) * 32)
        crop_size = max(crop_size, min_size)

        nx1 = int(max(0, min(fw - crop_size, cx_raw - crop_size / 2.0)))
        ny1 = int(max(0, min(fh - crop_size, cy_raw - crop_size / 2.0)))

        # 2. 物理裁剪
        crop_img = frame_rgb[ny1:ny1 + crop_size, nx1:nx1 + crop_size]

        if crop_img.size == 0:
            return False, 0.0

        # 3. YOLO 推理 (imgsz 建议设为 160)
        results = model.predict(crop_img, imgsz=160, conf=0.25, verbose=False)

        if len(results[0].boxes) > 0:
            self.is_drone_confirmed = True
            self.yolo_score = results[0].boxes.conf[0].item()
            self.negative_count = 0  # 验证成功，重置失败计数
            # print(f"ID:{self.track_id} 检测成功, 置信度: {self.yolo_score:.2f}")
            return True, self.yolo_score
        else:
            # 只有当 SUTrack 已经起批时，才累加失败计数
            if self.st_engine is not None:
                self.negative_count += 1

            # 为了方便观察，我们将 RGB 转回 BGR 再保存
            save_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

            # 文件名包含：ID、当前时间戳、负样计数
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ID{self.track_id}_{timestamp}_neg{self.negative_count}.jpg"
            save_full_path = os.path.join(DEBUG_SAVE_PATH, filename)
            # 可选：在图中画一个红圈表示 SOT 预测的中心，方便看是不是框歪了
            # 这里的坐标是相对于切片的中心
            local_cx = int(cx_raw - nx1)
            local_cy = int(cy_raw - ny1)
            cv2.circle(save_img, (local_cx, local_cy), 5, (0, 0, 255), -1)
            cv2.imwrite(save_full_path, save_img)
            # print(f"--- [Debug] 保存失败切片: {save_full_path}")
            return False, 0.0

    def mark_missed(self,step=1):
        self.misses += step
        self.hit_window.append(False)
        if len(self.hit_window) > WINDOW_SIZE: self.hit_window.pop(0)

class TrackManager:
    def __init__(self,sutrack_params=None,max_confirmed = 10):
        self.tracks = []
        self.next_id = 1
        self.sutrack_params = sutrack_params
        self.match_radius = 60  # 匹配像素半径
        # --- 新增：限制最大 SUTrack 数量 ---
        self.max_confirmed = max_confirmed

        # 新增：加载你训练出的最佳 YOLO 权重
        print("--- [System] 加载 YOLO 验证模型 ---")
        self.yolo_model = YOLO("./UAV_Custom_Training/blanced_custom_first_training3/weights/best.pt")
        print("--- [System] 全局预加载 SUTrack 网络权重 ---")
        self.shared_net = build_sutrack(sutrack_params.cfg)
        self.shared_net.load_state_dict(torch.load(sutrack_params.checkpoint)['net'])
        self.shared_net.cuda().eval()

    def _worker_deep_track(self, confirmed_tracks, frame_rgb):
        """GPU 密集型：SUTrack 自主更新"""
        for t in confirmed_tracks:
            t.update_by_sutrack(frame_rgb,self.yolo_model)

    def _worker_cv_discovery(self, pending_tracks, confirmed_tracks,detections, frame_rgb):

        """
        CPU 密集型：从检测点中寻找/验证新目标
            完整的追踪管理逻辑：
            1. 空间排斥：SUTrack 覆盖的点直接剔除
            2. 任务分流：Pending 航迹与剩余点匹配
            3. 新目标发现：起批剩余的高质量点
        """
        # 1. 预处理：区分已确认(SUTrack)和待定(Pending)航迹
        # --- 第一步：空间排斥 (核心重构点) ---
        # 剔除掉那些已经落在 SUTrack 跟踪范围内的检测点，防止重复起批
        clean_detections = []
        for det in detections:
            d_box = det['box']
            d_cx, d_cy = d_box[0] + d_box[2] / 2, d_box[1] + d_box[3] / 2
            is_covered = False
            for ct in confirmed_tracks:
                ct_cx, ct_cy = ct.box[0] + ct.box[2] / 2, ct.box[1] + ct.box[3] / 2
                dist = np.hypot(d_cx - ct_cx, d_cy - ct_cy)
                # 禁区半径：建议为目标最大边的 1.5 倍，最小不低于 40 像素
                exclusion_radius = max(max(ct.box[2], ct.box[3]) * 1.5, 40)
                if dist < exclusion_radius:
                    is_covered = True
                    break

            if not is_covered:
                clean_detections.append(det)
        # print("clean_detections",len(clean_detections))
        # 初始检测逻辑：如果既无待定航迹也无干净检测点，直接清理后返回
        if not pending_tracks and not clean_detections:
            # self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]
            return

        # --- 第二步：待定航迹的状态预测 ---
        for t in pending_tracks:
            pred = t.kf.predict()
            t.cached_predict = (pred[0][0], pred[1][0])

        # 如果没有干净的检测点，所有待定航迹标记丢失
        if not clean_detections:
            for t in pending_tracks:
                t.mark_missed()
            # self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]
            return

        # --- 第三步：代价矩阵计算 (针对 Pending 航迹) ---
        num_p = len(pending_tracks)
        num_d = len(clean_detections)

        # 限制检测点数量防止计算爆炸
        if num_d > 10:
            clean_detections = sorted(clean_detections, key=lambda x: x.get('score', 0), reverse=True)[:10]
            num_d = 10

        t_preds = np.array([t.get_predict() for t in pending_tracks], dtype=np.float32).reshape(num_p, 2)
        d_boxes = np.array([d['box'] for d in clean_detections], dtype=np.float32).reshape(num_d, 4)
        d_centers = d_boxes[:, :2] + d_boxes[:, 2:] / 2

        # 1. 基础距离矩阵
        cost_matrix = cdist(t_preds, d_centers, metric='euclidean')

        # 2. 批量特征提取用于代价优化
        t_boxes = np.array([t.box for t in pending_tracks], dtype=np.float32).reshape(num_p, 4)
        t_areas = t_boxes[:, 2] * t_boxes[:, 3]
        t_aspects = t_boxes[:, 2] / (t_boxes[:, 3] + 1e-6)
        t_vels = np.array([t.velocity for t in pending_tracks], dtype=np.float32).reshape(num_p, 2)
        t_prev_centers = t_boxes[:, :2] + t_boxes[:, 2:] / 2

        d_areas = d_boxes[:, 2] * d_boxes[:, 3]
        d_aspects = d_boxes[:, 2] / (d_boxes[:, 3] + 1e-6)

        # 3. 代价优化循环
        for i in range(num_p):
            # 面积与长宽比惩罚
            area_ratios = np.maximum(t_areas[i] / (d_areas + 1e-6), d_areas / (t_areas[i] + 1e-6))
            aspect_ratios = np.maximum(t_aspects[i] / (d_aspects + 1e-6), d_aspects / (t_aspects[i] + 1e-6))
            max_area_val = np.maximum(t_areas[i], d_areas)

            penalty_mask = ((max_area_val < 64) & ((area_ratios > 4.0) | (aspect_ratios > 3.0))) | \
                           ((max_area_val >= 64) & ((area_ratios > 2.5) | (aspect_ratios > 2.5)))
            cost_matrix[i, penalty_mask] += 300

            # 扇形搜索奖励/惩罚 (运动趋势一致性)
            t_vel = t_vels[i]
            t_speed = np.linalg.norm(t_vel)
            if t_speed > 2.5:
                displacements = d_centers - t_prev_centers[i]
                c_speeds = np.linalg.norm(displacements, axis=1)
                moving_mask = (c_speeds > 2.5)
                if np.any(moving_mask):
                    v_unit = t_vel / t_speed
                    d_units = displacements / (c_speeds[:, np.newaxis] + 1e-6)
                    cos_sim = np.sum(v_unit * d_units, axis=1)
                    # 夹角 45 度以内奖励，以外惩罚
                    in_sector = cos_sim > 0.707
                    cost_matrix[i, moving_mask & in_sector] *= 0.85
                    out_mask = moving_mask & (~in_sector)
                    cost_matrix[i, out_mask] += 150 * (1.0 - cos_sim[out_mask])

        # 极近距离保底信任
        cost_matrix[cost_matrix < 10.0] *= 0.6

        # --- 第四步：匈牙利算法匹配 ---
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_p, matched_d = set(), set()

        for p_idx, d_idx in zip(row_ind, col_ind):
            # 动态搜索半径：随丢失帧数增加而扩大
            dynamic_radius = MATCH_RADIUS * (1 + pending_tracks[p_idx].misses * 0.15)
            if cost_matrix[p_idx, d_idx] < dynamic_radius:
                # 更新 Pending 航迹，该方法内部应包含转正判定逻辑
                pending_tracks[p_idx].update_by_cv(clean_detections[d_idx]['box'], self, frame_rgb)
                matched_p.add(p_idx)
                matched_d.add(d_idx)

        # --- 第五步：善后处理 (未匹配标记与新起批) ---
        # 1. 未匹配的 Pending 标记丢失
        for i in range(num_p):
            if i not in matched_p:
                pending_tracks[i].mark_missed()

        # 2. 依然没被匹配的“干净”检测点 -> 开启新航迹起批
        for j in range(num_d):
            if j not in matched_d:
                det = clean_detections[j]
                score = det.get('score', 0)
                if score < 4.5: continue  # 起批门槛
                d_cx, d_cy = d_centers[j]
                # 空间去重：确保不和现有的任何待定目标靠得太近
                is_redundant = any(np.hypot(d_cx - (p.box[0] + p.box[2] / 2),
                                            d_cy - (p.box[1] + p.box[3] / 2)) < 20 for p in pending_tracks)
                if not is_redundant:
                    # 传入 self.tracks.append 开启新考察期
                    self.tracks.append(Track(det['box']))

        # 最终：统一清理失踪过久的所有目标
        # self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]

    def update(self, detections, frame_rgb):
        # 1. 任务分流
        confirmed, pending = [], []
        for t in self.tracks:
            if t.st_engine is not None:
                confirmed.append(t)
            else:
                pending.append(t)

        # 2. 多线程并行执行
        # 注意：t1 里的任务现在包含了：SUTrack跟踪 + 必要时的YOLO纠偏
        t1 = threading.Thread(target=self._worker_deep_track, args=(confirmed, frame_rgb))
        t2 = threading.Thread(target=self._worker_cv_discovery, args=(pending,confirmed, detections, frame_rgb))

        t1.start()
        t2.start()
        t1.join()
        t2.join()


        for t in self.tracks:
            # negative_count 在 update_by_sutrack 或 verify_by_yolo 中累加
            if t.negative_count >= Neg_count:
                print(f"--- [ID:{t.track_id}] 动态清理：YOLO 持续未检测到目标，判定为杂波 ---")
                t.st_engine = None  # 销毁引擎，使其从 Confirmed 降级
                t.is_confirmed = False
                t.is_drone_confirmed = False
                # t.misses = 999  # 强制让它在下一步清理中被删除
         # 唯一的清理入口：根据身份执行差异化清理 ---
        self.tracks = [
            t for t in self.tracks
            if (t.st_engine is not None and t.misses <= CONFIRMED_LIMIT) or
               (t.st_engine is None and t.misses <= PENDING_LIMIT)
        ]
        # --- 新增：盲跑状态监控 ---
        # active_confirmed = [t for t in self.tracks if t.st_engine is not None]
        # blind_count = sum(1 for t in active_confirmed if t.misses > 0)

        # if len(active_confirmed) > 0:
        #     print(
        #         f"[Monitor] 已锁定:{len(active_confirmed)} | 盲跑中:{blind_count} | 待定中:{len(self.tracks) - len(active_confirmed)}")
        #     # 如果有目标即将超时，打印警告
        #     for t in active_confirmed:
        #         if t.misses > (CONFIRMED_LIMIT * 0.8):  # 比如超过 48 帧
        #             print(f"  ! Warning: ID {t.track_id} 接近生命极限 (Misses: {t.misses})")
        #


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
def cluster_detections_initial(candidates, cluster_dist=10):
    if not candidates:
        return []

    boxes = np.array([c['box'] for c in candidates])
    areas = np.array([c['area'] for c in candidates])
    n = len(candidates)

    if n == 1:
        return candidates

    centers = np.column_stack([
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2
    ])

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    adj = dist_matrix < cluster_dist

    visited = np.zeros(n, dtype=bool)
    clustered_results = []

    for i in range(n):
        if visited[i]:
            continue

        component = []
        queue = deque([i])
        visited[i] = True

        while queue:
            curr = queue.popleft()
            component.append(curr)  # 【修复1】：将当前点加入组件列表

            # 找到所有未访问的邻居
            neighbors = np.where(adj[curr] & ~visited)[0]
            for nb in neighbors:
                visited[nb] = True
                queue.append(nb)

        # 【修复2】：增加防御性判断，确保组件不为空
        if len(component) > 0:
            comp_boxes = boxes[component]
            x1 = np.min(comp_boxes[:, 0])
            y1 = np.min(comp_boxes[:, 1])
            x2 = np.max(comp_boxes[:, 0] + comp_boxes[:, 2])
            y2 = np.max(comp_boxes[:, 1] + comp_boxes[:, 3])

            clustered_results.append({
                'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                'area': int(np.sum(areas[component])),
                'fragments': len(component)
            })

    return clustered_results

# def cluster_detections_initial(candidates, cluster_dist=10):
#     """
#     高效初始聚类：将距离较近的检测碎点合并为完整目标。
#     """
#     if not candidates:
#         return []
#
#     # 1. 提取 Numpy 数组以实现加速计算
#     boxes = np.array([c['box'] for c in candidates])  # [N, 4] -> x, y, w, h
#     areas = np.array([c['area'] for c in candidates])  # [N]
#     n = len(candidates)
#
#     if n == 1:
#         return candidates
#
#     # 计算每个碎点的中心位置
#     centers = np.column_stack([
#         boxes[:, 0] + boxes[:, 2] / 2,
#         boxes[:, 1] + boxes[:, 3] / 2
#     ])
#
#     # 2. 向量化计算距离矩阵 (N, N)
#     # 利用 Numpy 广播特性，一行代码算出所有点两两之间的距离
#     diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
#     dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
#
#     # 3. 构建邻接矩阵：距离小于阈值的点视为“相连”
#     adj = dist_matrix < cluster_dist
#
#     visited = np.zeros(n, dtype=bool)
#     clustered_results = []
#
#     # 4. 寻找连通分量 (把连在一起的碎点打成一包)
#     for i in range(n):
#         if visited[i]:
#             continue
#
#         # 使用简单的 BFS 找到所有连通的点
#         component = []
#         queue = [i]
#         visited[i] = True
#         while queue:
#             curr = queue.pop(0)
#             neighbors = np.where(adj[curr] & ~visited)[0]
#             for nb in neighbors:
#                 visited[nb] = True
#                 queue.append(nb)
#
#         # 5. 聚合：计算包围所有碎点的“最大外接矩形”和“总面积”
#         comp_boxes = boxes[component]
#         x1 = np.min(comp_boxes[:, 0])
#         y1 = np.min(comp_boxes[:, 1])
#         x2 = np.max(comp_boxes[:, 0] + comp_boxes[:, 2])
#         y2 = np.max(comp_boxes[:, 1] + comp_boxes[:, 3])
#
#         clustered_results.append({
#             'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
#             'area': int(np.sum(areas[component])),
#             'fragments': len(component)  # 记录碎点数，这对后续评分很有用
#         })
#
#     return clustered_results

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

    if num > 1:
        # 获取除了背景（index 0）以外的所有数据
        valid_stats = stats[1:]
        ws = valid_stats[:, 2]
        hs = valid_stats[:, 3]
        areas = valid_stats[:, 4]

        # 向量化过滤：一次性计算所有纵横比和面积
        aspect_ratios = ws / hs
        mask = (aspect_ratios > 0.3) & (aspect_ratios < 3.0) & (areas > 2) & (areas < 600)

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
    # 在循环开始前（run_system 初始化部分）
    # cv2.namedWindow("Debug Board", cv2.WINDOW_NORMAL)  # 允许手动拉大窗口
    # cv2.moveWindow("Debug Board", 100, 100)  # 强制移动到屏幕可见位置
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
        'CLUSTER_DIST': 30,
        'ALPHA': 0.3,
        'W_MOTION': 1.5,
        'W_AREA': 1.2,
        'W_SPATIAL': 1.5,
        'MOTION_FLOOR': 1.0,
        'BUFFER_PIXELS': 0,
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        # print(f"current_frame {frame_count}")
        # 每一帧开始时，初始化调试画布
        canvas = frame.copy()

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
                # A. 计算裁剪边界：找到天际线最深的点，增加 像素缓冲区
                y_cutoff = int(np.max(sky_line)+2)
                y_cutoff = min(fh, y_cutoff)  # 防止越界

                # B. 实施物理裁剪 (生成新的更小的内存矩阵)
                # 只有这里切了，后面的 cvtColor, CLAHE, Blackhat 才会提速
                roi_frame = frame[0:y_cutoff, :]
                roi_sky_mask = sky_mask[0:y_cutoff, :]
                # C. 调用检测函数：传入裁剪后的图
                # 此时 detect_on_crop 内部处理的像素点大幅减少
                # cands = detect_on_crop(roi_frame, roi_sky_mask, offset_x=0, offset_y=0)
                # ============ 【物理裁剪修改结束】 ============
                # 暴力提取原始碎点 (此时 detect_on_crop 只管抓点，不管分数)
                raw_cands = detect_on_crop(roi_frame, roi_sky_mask, offset_x=0, offset_y=0)
                # print("raw_cands",len(raw_cands))
                # # --- B. 调试第一层：原始碎点 (Blue) ---
                # raw_cands = detect_on_crop(roi_frame, roi_sky_mask, offset_x=0, offset_y=0)
                # for cand in raw_cands:
                #     x, y, w, h = cand['box']
                #     cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 1)

                # 【核心重构：先聚类】

                # 使用我们之前写的高效版聚类，将碎点合成为疑似目标
                clustered_cands = cluster_detections_initial(raw_cands, cluster_dist=20)

                # # --- C. 调试第二层：物理聚类 (Yellow) ---
                # # 先聚类，减少后续光流验证的次数
                # clustered_cands = cluster_detections_initial(raw_cands, cluster_dist=20)
                # for cand in clustered_cands:
                #     x, y, w, h = cand['box']
                #     cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #     cv2.putText(canvas, f"f:{cand.get('fragments', 1)}", (x, y - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                if len(clustered_cands) > 0:
                    # --- 向量化优化开始 ---
                    # 1. 整理所有待检测点的中心坐标 (N, 1, 2)
                    pts_list = []
                    for cand in clustered_cands:
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
                    for i, cand in enumerate(clustered_cands):
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
                                # 关键：使用聚类后的总面积进行面积加分
                                area_val = cand['area']
                                area_part = cfg['W_AREA'] * np.log1p(area_val + 1)
                                # # 额外加分项：由多个碎片聚成的目标置信度更高
                                # fragment_bonus = 0.5 if cand.get('fragment_count', 1) > 1 else 0
                                score = motion_part + area_part + spatial_bonus
                                if score > 2.0:
                                    cand['score'] = max(0.01, score)
                                    validated_with_score.append(cand)

        # 4. 聚类与追踪
        final_detections = cluster_scored_detections(validated_with_score, cluster_dist=cfg['CLUSTER_DIST'])
        # # --- 4. 调试第三层：最终聚合与评分结果 (Green) ---
        # final_detections = cluster_scored_detections(validated_with_score, cluster_dist=cfg['CLUSTER_DIST'])
        # for cand in final_detections:
        #     x, y, w, h = cand['box']
        #     s = cand['score']
        #     cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.putText(canvas, f"S:{s:.1f}", (x, y + h + 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # # --- 5. 显示与保存 ---
        # if canvas is not None:
        #     # 缩小画面可以极大地提高远程传输成功率，防止黑屏
        #     display_size = (960, 540)
        #     debug_small = cv2.resize(canvas, display_size)
        #
        #     cv2.imshow("Debug Board", debug_small)
        #
        #     # 远程环境下 waitKey(1) 有时太快，尝试加大到 30 (约 30ms)
        #     # 如果按下 'q' 键则退出
        #     key = cv2.waitKey(1000) & 0xFF
        #     if key == ord('q'):
        #         break
        # --- 核心修改：调用多线程 update ---

        tracker.update(final_detections, frame_rgb)

        # 绘制天际线
        for c in range(fw - 1):
            cv2.line(frame, (c, sky_line[c]), (c + 1, sky_line[c + 1]), (0, 255, 255), 1)

        active_count = 0
        for t in tracker.tracks:
            x, y, w, h = [int(v) for v in t.box]
            # 1. 已起批目标 (Confirmed Tracks)
            if t.st_engine is not None:
                active_count += 1
                # --- 核心：根据 YOLO 验证状态区分颜色 ---
                if t.is_drone_confirmed:
                    # 【状态：绿框】SUTrack 锁定 + YOLO 确认是无人机
                    color = (0, 255, 0)
                    status_text = f"UAV ID:{t.track_id} {t.yolo_score:.2f}"
                else:
                    # 【状态：深粉色】SUTrack 正在跟踪，但 YOLO 还没认出来（或正在考察）
                    # 如果 negative_count 很高，你会看到它在被清理前的最后挣扎
                    color = (147, 20, 255)
                    status_text = f"CHECK ID:{t.track_id} Neg:{t.negative_count}"
                if t.misses > 0:
                    color = (128, 128, 128)
                    status_text += f" LOST:{t.misses}"
                # 绘图
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(frame, status_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # 绘制轨迹 (SUTrack 轨迹用黄色虚线感)
                if len(t.trace) > 2:
                    for i in range(1, len(t.trace)):
                        cv2.line(frame, t.trace[i - 1], t.trace[i], (0, 255, 255), 1)

            # 2. 待确认目标 (Pending Tracks)
            else:
                # 【状态：黄色窄框】传统 CV 刚发现的目标，还在攒“命中率”，准备起批
                color = (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                # 可以在框上方标个 P 代表 Pending
                cv2.putText(frame, "P", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

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


import os
from pathlib import Path
import torch

def batch_run():
    # 1. 配置路径 (基于 sx/SUTrack 的相对路径)
    video_dir = Path("test_videos/Demo")
    result_dir = Path("results/Demo")

    # 2. 检查并创建结果目录
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- 已创建保存目录: {result_dir} ---")

    # 3. 筛选视频文件
    # rglob 会搜索目录下所有匹配的视频
    video_list = sorted([
        f for f in video_dir.iterdir()
        if f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']
    ])

    if not video_list:
        print(f"错误: 在 {video_dir.absolute()} 下未找到视频文件！")
        return

    print(f"共发现 {len(video_list)} 个视频，准备开始批量推理...")

    # 4. 循环调用 run_system
    for video_path in video_list:
        # 构建保存路径 (例如 results/2026.1.15_2/test.mp4)
        save_path = str(result_dir / video_path.name)

        print(f"\n>>> 正在处理: {video_path.name}")

        try:
            # 这里的 run_system 是你之前定义的函数
            run_system(
                str(video_path),
                save_path,
                sutrack_cfg="sutrack_b384"
            )

            # 每跑完一个视频，清理一次显存碎片，保证长时间运行稳定性
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"!!! 视频 {video_path.name} 处理异常: {e}")
            continue

    print("\n[Done] 所有视频已处理完毕。")
if __name__ == "__main__":
    # --- 全局配置 ---
    VIDEO_PATH = './test_videos/2026.1.15_2/DJI_20260115174111_0002_Z.MP4'
    # SAVE_PATH = "./results/2026.1.15_2/DJI_20260115174111_0002_Z.MP4"
    # run_system(VIDEO_PATH, SAVE_PATH,sutrack_cfg="sutrack_b224")
    batch_run()