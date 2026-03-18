import os
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================

SOURCE_AREA = "/home/poker/workspace/11_third_projs/data/train_data/02_version/c4/Area_1"
OUTPUT_ROOT = "/home/poker/workspace/11_third_projs/data/train_data/02_version/c4/c4_dataset"

TRAIN_RATIO = 0.85

# c4 下 c2 / c3 占比小，但通常位于高曲率区域
RARE_CLASSES = [2, 3]

# 骨骼 bbox 大约 40~60 mm。
# 训练时已经有 CenterShift / 旋转 / 缩放，因此离线增强更重要的是：
# 1) 挑更有信息量的位置
# 2) 减少重复 patch
# 3) 做多尺度上下文，而不是重复几何抖动
RARE_RADII = (2.2, 2.8, 3.4)
BOUNDARY_RADII = (2.8, 3.4, 4.0)
HARD_RADII = (3.6, 4.2)

BOUNDARY_RADIUS = 1.0
BOUNDARY_RADIUS_STRICT = 0.5

MIN_PATCH_POINTS = 150
MIN_RARE_POINTS = 24
MIN_HARD_CLASS_POINTS = 48
MIN_BOUNDARY_CLASS_POINTS = 20
MIN_BOUNDARY_CLASS_RATIO = 0.08

RARE_CLASS_PATCH_MAX = {
    2: 6,
    3: 6,
}
BOUNDARY_PATCH_MAX = 8
BOUNDARY_STRICT_PATCH_MAX = 4
HARD_PATCH_MAX = 6

RARE_CENTER_DISTANCE = 2.0
BOUNDARY_CENTER_DISTANCE = 2.5
HARD_CENTER_DISTANCE = 3.5

CURVATURE_K = 24

NUM_WORKERS = 24
GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


# ===============================
# load room
# ===============================

def load_room(room_path):

    coord = np.load(room_path / "coord.npy")
    normal = np.load(room_path / "normal.npy")
    label = np.load(room_path / "segment.npy")

    return coord, normal, label


# ===============================
# patch helpers
# ===============================

def build_room_rng(room):

    seed = (zlib.crc32(room.encode("utf-8")) + GLOBAL_SEED) & 0xFFFFFFFF
    return random.Random(seed)


def crop_patch(coord, center, radius):

    diff = coord - center
    dist2 = np.sum(diff * diff, axis=1)

    return dist2 < radius * radius


def choose_mask_by_radii(coord, center, radii, rng):

    candidates = []

    for radius in radii:

        mask = crop_patch(coord, center, radius)

        if int(mask.sum()) >= MIN_PATCH_POINTS:
            candidates.append((radius, mask))

    if not candidates:
        return None, None

    radius, mask = rng.choice(candidates)

    return radius, mask


def is_far_enough(point, existing_points, min_distance):

    if not existing_points:
        return True

    for other in existing_points:
        if np.linalg.norm(point - other) < min_distance:
            return False

    return True


def select_diverse_centers(
    coord,
    candidate_idx,
    scores,
    max_count,
    min_distance,
    rng,
    existing_centers=None,
):

    candidate_idx = np.asarray(candidate_idx, dtype=np.int32)

    if len(candidate_idx) == 0 or max_count <= 0:
        return []

    candidate_idx = np.unique(candidate_idx)

    decorated = []

    for idx in candidate_idx.tolist():
        score = float(scores[idx]) if scores is not None else 0.0
        decorated.append((score, rng.random(), idx))

    decorated.sort(reverse=True)

    selected = []
    selected_points = []

    if existing_centers is None:
        existing_centers = []

    for _, _, idx in decorated:

        point = coord[idx]

        if not is_far_enough(point, existing_centers, min_distance):
            continue

        if not is_far_enough(point, selected_points, min_distance):
            continue

        selected.append(idx)
        selected_points.append(point)

        if len(selected) >= max_count:
            break

    return selected


# ===============================
# geometry / boundary signal
# ===============================

def compute_curvature_score(coord, normal):

    if len(coord) <= 2:
        return np.zeros(len(coord), dtype=np.float32)

    k = min(CURVATURE_K, len(coord))
    tree = cKDTree(coord)
    _, idx = tree.query(coord, k=k)

    if k == 1:
        idx = idx[:, None]

    neighbor_normals = normal[idx]
    center_normals = normal[:, None, :]
    cosine = np.sum(neighbor_normals * center_normals, axis=2)
    cosine = np.clip(cosine, -1.0, 1.0)

    # 法向变化越大，分数越高，可近似看作高曲率区域
    score = np.mean(1.0 - cosine, axis=1)

    return score.astype(np.float32)


def find_boundary_points(coord, label, tree, radius):

    boundary = []

    for i in range(len(coord)):

        idx = tree.query_ball_point(coord[i], radius)

        if len(idx) < 5:
            continue

        if len(np.unique(label[idx])) > 1:
            boundary.append(i)

    return np.asarray(boundary, dtype=np.int32)


def build_rare_boundary_bonus(coord, label, tree, candidate_idx, radius):

    bonus = np.zeros(len(label), dtype=np.float32)

    for idx in candidate_idx:

        neighbor_idx = tree.query_ball_point(coord[idx], radius)
        neighbor_labels = label[neighbor_idx]

        has_rare = np.any(np.isin(neighbor_labels, RARE_CLASSES))
        has_non_rare = np.any(~np.isin(neighbor_labels, RARE_CLASSES))

        if has_rare and has_non_rare:
            bonus[idx] = 0.5
        elif has_rare:
            bonus[idx] = 0.25

    return bonus


def is_valid_boundary_patch(labels_in_patch):

    unique, counts = np.unique(labels_in_patch, return_counts=True)

    if len(unique) < 2:
        return False

    min_count = int(counts.min())
    min_ratio = float(counts.min() / counts.sum())

    return (
        min_count >= MIN_BOUNDARY_CLASS_POINTS
        and min_ratio >= MIN_BOUNDARY_CLASS_RATIO
    )


# ===============================
# save
# ===============================

def merge_and_save(coord, normal, label, out_file):

    merged = np.concatenate(
        [
            coord.astype(np.float32),
            normal.astype(np.float32),
            label.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    )

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, merged)


def save_patch(coord, normal, label, name, area):

    out_dir = OUTPUT_ROOT_PATH / area / name
    merge_and_save(coord, normal, label, out_dir / "data.npy")


def try_save_patch(
    coord,
    normal,
    label,
    center_idx,
    radii,
    name,
    area,
    rng,
    min_target_points=0,
    target_classes=None,
    require_boundary=False,
    require_class=None,
):

    radius, mask = choose_mask_by_radii(coord, coord[center_idx], radii, rng)

    if mask is None:
        return False

    label_patch = label[mask]

    if target_classes is not None:
        target_count = int(np.isin(label_patch, target_classes).sum())
        if target_count < min_target_points:
            return False

    if require_boundary and not is_valid_boundary_patch(label_patch):
        return False

    if require_class is not None:
        class_count = int((label_patch == require_class).sum())
        if class_count < MIN_HARD_CLASS_POINTS:
            return False

    save_patch(
        coord[mask],
        normal[mask],
        label_patch,
        name,
        area,
    )

    return True


# ===============================
# process room
# ===============================

def process_room(room):

    room_path = SOURCE_PATH / room
    coord, normal, label = load_room(room_path)

    room_rng = build_room_rng(room)
    tree = cKDTree(coord)
    curvature = compute_curvature_score(coord, normal)

    total = 0
    used_centers = []

    # =====================
    # rare patch
    # 按类别分别取，避免 c2 / c3 其中一类被另一类淹没
    # =====================

    for rare_class, patch_max in RARE_CLASS_PATCH_MAX.items():

        rare_idx = np.where(label == rare_class)[0]

        if len(rare_idx) == 0:
            continue

        rare_scores = curvature.copy()
        rare_scores[rare_idx] += 1.0

        selected = select_diverse_centers(
            coord=coord,
            candidate_idx=rare_idx,
            scores=rare_scores,
            max_count=patch_max,
            min_distance=RARE_CENTER_DISTANCE,
            rng=room_rng,
            existing_centers=used_centers,
        )

        saved = 0

        for idx in selected:
            ok = try_save_patch(
                coord=coord,
                normal=normal,
                label=label,
                center_idx=idx,
                radii=RARE_RADII,
                name=f"{room}_rare_c{rare_class}_{saved}",
                area="Area_patch_rare",
                rng=room_rng,
                min_target_points=MIN_RARE_POINTS,
                target_classes=[rare_class],
            )

            if ok:
                used_centers.append(coord[idx])
                total += 1
                saved += 1

    # =====================
    # boundary patch
    # 优先 rare class 参与的边界，同时控制中心间距，减少重复 patch
    # =====================

    boundary_idx = find_boundary_points(coord, label, tree, radius=BOUNDARY_RADIUS)

    if len(boundary_idx) > 0:

        boundary_scores = curvature + build_rare_boundary_bonus(
            coord=coord,
            label=label,
            tree=tree,
            candidate_idx=boundary_idx,
            radius=BOUNDARY_RADIUS,
        )

        selected = select_diverse_centers(
            coord=coord,
            candidate_idx=boundary_idx,
            scores=boundary_scores,
            max_count=BOUNDARY_PATCH_MAX,
            min_distance=BOUNDARY_CENTER_DISTANCE,
            rng=room_rng,
            existing_centers=used_centers,
        )

        saved = 0

        for idx in selected:
            ok = try_save_patch(
                coord=coord,
                normal=normal,
                label=label,
                center_idx=idx,
                radii=BOUNDARY_RADII,
                name=f"{room}_boundary_{saved}",
                area="Area_patch_boundary",
                rng=room_rng,
                require_boundary=True,
            )

            if ok:
                used_centers.append(coord[idx])
                total += 1
                saved += 1

    # =====================
    # strict boundary patch
    # 更贴近界面，用更小邻域找中心，再用中等 patch 看完整过渡
    # =====================

    boundary_strict_idx = find_boundary_points(
        coord,
        label,
        tree,
        radius=BOUNDARY_RADIUS_STRICT,
    )

    if len(boundary_strict_idx) > 0:

        strict_scores = curvature + build_rare_boundary_bonus(
            coord=coord,
            label=label,
            tree=tree,
            candidate_idx=boundary_strict_idx,
            radius=BOUNDARY_RADIUS,
        )

        selected = select_diverse_centers(
            coord=coord,
            candidate_idx=boundary_strict_idx,
            scores=strict_scores,
            max_count=BOUNDARY_STRICT_PATCH_MAX,
            min_distance=BOUNDARY_CENTER_DISTANCE,
            rng=room_rng,
            existing_centers=used_centers,
        )

        saved = 0

        for idx in selected:
            ok = try_save_patch(
                coord=coord,
                normal=normal,
                label=label,
                center_idx=idx,
                radii=BOUNDARY_RADII,
                name=f"{room}_boundary_strict_{saved}",
                area="Area_patch_boundary",
                rng=room_rng,
                require_boundary=True,
            )

            if ok:
                used_centers.append(coord[idx])
                total += 1
                saved += 1

    # =====================
    # hard patch
    # 不再从所有非 0 类随机采样，而是优先 class1 非边界内部区域，
    # 补充更大上下文，降低训练只记住 rare/boundary 局部形状的风险
    # =====================

    boundary_mask = np.zeros(len(label), dtype=bool)
    boundary_mask[boundary_idx] = True if len(boundary_idx) > 0 else False

    hard_idx = np.where((label == 1) & (~boundary_mask))[0]

    if len(hard_idx) > 0:

        selected = select_diverse_centers(
            coord=coord,
            candidate_idx=hard_idx,
            scores=None,
            max_count=HARD_PATCH_MAX,
            min_distance=HARD_CENTER_DISTANCE,
            rng=room_rng,
            existing_centers=used_centers,
        )

        saved = 0

        for idx in selected:
            ok = try_save_patch(
                coord=coord,
                normal=normal,
                label=label,
                center_idx=idx,
                radii=HARD_RADII,
                name=f"{room}_hard_{saved}",
                area="Area_patch_hard",
                rng=room_rng,
                require_class=1,
            )

            if ok:
                used_centers.append(coord[idx])
                total += 1
                saved += 1

    return total


# ===============================
# main
# ===============================

def main():

    global SOURCE_PATH
    global OUTPUT_ROOT_PATH

    SOURCE_PATH = Path(SOURCE_AREA)
    OUTPUT_ROOT_PATH = Path(OUTPUT_ROOT)

    rooms = [
        room for room in os.listdir(SOURCE_AREA)
        if os.path.isdir(os.path.join(SOURCE_AREA, room))
    ]

    print("rooms:", len(rooms))

    random.shuffle(rooms)

    train_num = int(len(rooms) * TRAIN_RATIO)
    train_rooms = rooms[:train_num]
    val_rooms = rooms[train_num:]

    # 将源房间合并为单文件 data.npy 写入输出
    print("copy dataset (merge to data.npy)")

    def copy_room_merged(room, area_name):
        src = SOURCE_PATH / room
        dst_dir = OUTPUT_ROOT_PATH / area_name / room
        coord, normal, label = load_room(src)
        merge_and_save(coord, normal, label, dst_dir / "data.npy")

    all_copy_tasks = (
        [(room, "Area_1") for room in train_rooms]
        + [(room, "Area_5") for room in val_rooms]
    )

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        list(
            tqdm(
                ex.map(lambda x: copy_room_merged(x[0], x[1]), all_copy_tasks),
                total=len(all_copy_tasks),
                desc="copy",
            )
        )

    print("generate patches")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        res = list(
            tqdm(
                ex.map(process_room, train_rooms),
                total=len(train_rooms),
                desc="rooms",
            )
        )

    print("total patches:", sum(res))


if __name__ == "__main__":

    main()