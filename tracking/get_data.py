from track_combine_test1 import *


class DynamicDatasetCollector:
    def __init__(self, project_name="UAV_Training_Data", padding_factor=2.5, min_size=128):
        self.root = project_name
        self.padding_factor = padding_factor
        self.min_size = min_size
        self.img_dir = os.path.join(self.root, "images")
        self.lbl_dir = os.path.join(self.root, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
        self.counter = 0

    def collect_frame(self, frame, tracks, frame_id, target_ids, neg_ids=None, video_prefix="vid"):
        fh, fw = frame.shape[:2]
        neg_ids = neg_ids or []  # 默认为空列表

        for t in tracks:
            # 只有当追踪 ID 在【正样本列表】或【负样本列表】中，且当前帧目标未丢失时才收集
            combined_all_ids = target_ids + neg_ids
            if t.track_id in combined_all_ids and t.misses <= 10:
                # --- 以下所有逻辑与你原始代码完全一致 ---
                tx, ty, tw, th = t.box
                cx_raw, cy_raw = tx + tw / 2.0, ty + th / 2.0

                side = max(tw, th) * self.padding_factor
                crop_size = int(np.ceil(side / 32) * 32)
                crop_size = max(crop_size, self.min_size)

                nx1 = int(max(0, min(fw - crop_size, cx_raw - crop_size / 2.0)))
                ny1 = int(max(0, min(fh - crop_size, cy_raw - crop_size / 2.0)))

                crop_img = frame[ny1:ny1 + crop_size, nx1:nx1 + crop_size]

                # 文件名规则 (如果是负样本，加上 _neg 标识方便你以后肉眼区分)
                suffix = "_neg" if t.track_id in neg_ids else ""
                file_base = f"{video_prefix}_f{frame_id:06d}_id{t.track_id}{suffix}_{self.counter}"

                # 保存图片
                cv2.imwrite(os.path.join(self.img_dir, f"{file_base}.jpg"), crop_img)

                # --- 修改写标签的逻辑 ---
                lbl_path = os.path.join(self.lbl_dir, f"{file_base}.txt")
                with open(lbl_path, 'w') as f:
                    if t.track_id in neg_ids:
                        # 如果是负样本，保持文件为空 (YOLO 要求的背景样本格式)
                        f.write("")
                    else:
                        # 正样本逻辑保持完全不变
                        new_cx = (cx_raw - nx1) / crop_size
                        new_cy = (cy_raw - ny1) / crop_size
                        new_bw = tw / crop_size
                        new_bh = th / crop_size
                        f.write(f"0 {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}\n")

                self.counter += 1


def run_system(video_path, collection_tasks=None, sutrack_cfg="sutrack_b224"):
    """
        极速采集版：针对多视频防重名和稳定性进行了深度优化
        """
    # 0. 自动提取视频文件名作为前缀
    video_abs_path = os.path.abspath(video_path)
    video_name_prefix = os.path.splitext(os.path.basename(video_abs_path))[0]

    # 1. 初始化
    collector = DynamicDatasetCollector(project_name="UAV_Training_Data")
    sut_params = parameters(sutrack_cfg)
    sut_params.debug = 0  # 提速

    start_time = time.time()
    # 强制使用 FFMPEG 插件，避免路径数字引起 OpenCV 误判
    cap = cv2.VideoCapture(video_abs_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_abs_path}")
        return

    fw, fh = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    print(f"🎬 开始处理视频: {video_name_prefix} | 预计: {total_frames} 帧")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
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

        # D. 核心采集逻辑
        if collection_tasks:
            pos_ids = []
            neg_ids = []
            for task in collection_tasks:
                if task["range"][0] <= frame_count <= task["range"][1]:
                    # 分别收集正负 ID
                    pos_ids.extend(task.get("ids", []))
                    neg_ids.extend(task.get("neg_ids", []))

            if pos_ids or neg_ids:
                collector.collect_frame(
                    frame,
                    tracker.tracks,
                    frame_count,
                    target_ids=list(set(pos_ids)),
                    neg_ids=list(set(neg_ids)),
                    video_prefix=video_name_prefix
                )

        # E. 进度反馈
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"进度: {frame_count}/{total_frames} | 速度: {frame_count / elapsed:.1f} FPS | 收集数: {collector.counter}")

        prev_gray = curr_gray.copy()

    cap.release()
    print(f"✅ 采集完成！数据保存在: {os.path.abspath(collector.root)}")

if __name__ == "__main__":
    run_system(
        video_path='./test_videos/2026.1.08/DJI_20260110144928_0011_W.MP4',
        collection_tasks=[
            # 正常采集 ID 1 (正样本)
            {"range": (145, 195), "ids": [3]},
            {"range": (242, 335), "ids": [3]},
            {"range": (421, 779), "ids": [3]},

            # {"range": (790, 828), "ids": [3]},
            # {"range": (1763, 2073), "ids": [3]},
            # {"range": (2821, 3034), "ids": [1,2,3,4]},

            # 同时采集 ID 2 (正样本) 和 ID 5 (比如是误报的干扰物，设为负样本)
            # {"range": (1302, 1355), "ids": [2], "neg_ids": [1,4]},
            # {"range": (1376, 1902), "ids": [2], "neg_ids": [1,3,4]},

            # 这一段只要 ID 3 的负样本
            {"range": (387, 400), "neg_ids": [3]},
            # {"range": (2380, 4878), "neg_ids": [1,2,3,4,5]},
            # {"range": (1225, 2700), "neg_ids": [2]},
            # {"range": (307, 1041), "neg_ids": [2,3]},
        ]
    )
