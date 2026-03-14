import os
import numpy as np
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================

SOURCE_AREA = "/home/poker/workspace/11_third_projs/data/train_data/01_version/c4/Area_1"

OUTPUT_ROOT = "/home/poker/workspace/11_third_projs/data/train_data/01_version/c4/c4_dataset"

TRAIN_RATIO = 0.85

RARE_CLASSES = [2,3]

PATCH_RADIUS = 3.0
BOUNDARY_RADIUS = 1.0
# 更小半径：只把「紧贴界面」的点当边界，用于多采一批边界 patch
BOUNDARY_RADIUS_STRICT = 0.5

MIN_PATCH_POINTS = 150
# 边界 patch 内至少 2 类，且每类点数不少于该值，保证 patch 内两侧都可见
MIN_BOUNDARY_CLASS_POINTS = 20

RARE_PATCH_MAX = 15
# 边界易错：提高边界 patch 数量
BOUNDARY_PATCH_MAX = 20
BOUNDARY_STRICT_PATCH_MAX = 10  # 从「严格边界」再采的 patch 数
HARD_PATCH_MAX = 5

NUM_WORKERS = 24

random.seed(42)


# ===============================
# load room
# ===============================

def load_room(room_path):

    coord = np.load(room_path/"coord.npy")
    normal = np.load(room_path/"normal.npy")
    label = np.load(room_path/"segment.npy")

    return coord,normal,label


# ===============================
# crop sphere
# ===============================

def crop_patch(coord,center,radius):

    dist = np.linalg.norm(coord-center,axis=1)

    return dist<radius


# ===============================
# boundary detection
# ===============================

def find_boundary_points(coord, label, radius=None):
    radius = radius if radius is not None else BOUNDARY_RADIUS
    tree = cKDTree(coord)
    boundary = []
    for i in range(len(coord)):
        idx = tree.query_ball_point(coord[i], radius)
        if len(idx) < 5:
            continue
        if len(set(label[idx])) > 1:
            boundary.append(i)
    return np.array(boundary)


def is_valid_boundary_patch(labels_in_patch):
    """边界 patch 需至少 2 类，且每类点数 >= MIN_BOUNDARY_CLASS_POINTS，避免单侧占满。"""
    unique, counts = np.unique(labels_in_patch, return_counts=True)
    if len(unique) < 2:
        return False
    return counts.min() >= MIN_BOUNDARY_CLASS_POINTS


# ===============================
# 合并并保存为单文件 data.npy（格式: [x,y,z, nx,ny,nz, segment] -> (N,7) float32）
# ===============================

def merge_and_save(coord, normal, label, out_file):
    """将 coord、normal、segment 合并为 (N,7) float32 并写入单个 npy 文件。"""
    merged = np.concatenate(
        [coord.astype(np.float32), normal.astype(np.float32), label.reshape(-1, 1).astype(np.float32)],
        axis=1,
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, merged)


def save_patch(coord, normal, label, name, area):
    out_dir = OUTPUT_ROOT_PATH / area / name
    merge_and_save(coord, normal, label, out_dir / "data.npy")


# ===============================
# process room
# ===============================

def process_room(room):

    room_path = SOURCE_PATH/room

    coord,normal,label = load_room(room_path)

    total=0

    # =====================
    # rare patch
    # =====================

    rare_idx=np.where(np.isin(label,RARE_CLASSES))[0]

    if len(rare_idx)>0:

        selected=random.sample(
            list(rare_idx),
            min(len(rare_idx),RARE_PATCH_MAX)
        )

        for i,idx in enumerate(selected):

            mask=crop_patch(coord,coord[idx],PATCH_RADIUS)

            if mask.sum()<MIN_PATCH_POINTS:
                continue

            save_patch(
                coord[mask],
                normal[mask],
                label[mask],
                f"{room}_rare_{i}",
                "Area_patch_rare"
            )

            total+=1


    # =====================
    # boundary patch（边界易错：多采 + 保证 patch 内两侧都有足够点）
    # =====================

    boundary_idx = find_boundary_points(coord, label)

    if len(boundary_idx) > 0:
        selected = random.sample(
            list(boundary_idx),
            min(len(boundary_idx), BOUNDARY_PATCH_MAX)
        )
        saved = 0
        for idx in selected:
            mask = crop_patch(coord, coord[idx], PATCH_RADIUS)
            if mask.sum() < MIN_PATCH_POINTS:
                continue
            if not is_valid_boundary_patch(label[mask]):
                continue
            save_patch(
                coord[mask], normal[mask], label[mask],
                f"{room}_boundary_{saved}", "Area_patch_boundary"
            )
            total += 1
            saved += 1

    # 严格边界：更小邻域内的界面点，再采一批 patch，强化边界
    boundary_strict_idx = find_boundary_points(coord, label, radius=BOUNDARY_RADIUS_STRICT)
    if len(boundary_strict_idx) > 0:
        selected = random.sample(
            list(boundary_strict_idx),
            min(len(boundary_strict_idx), BOUNDARY_STRICT_PATCH_MAX)
        )
        saved = 0
        for idx in selected:
            mask = crop_patch(coord, coord[idx], PATCH_RADIUS)
            if mask.sum() < MIN_PATCH_POINTS:
                continue
            if not is_valid_boundary_patch(label[mask]):
                continue
            save_patch(
                coord[mask], normal[mask], label[mask],
                f"{room}_boundary_strict_{saved}", "Area_patch_boundary"
            )
            total += 1
            saved += 1


    # =====================
    # hard patch
    # =====================

    hard_idx=np.where(label!=0)[0]

    if len(hard_idx)>0:

        selected=random.sample(
            list(hard_idx),
            min(len(hard_idx),HARD_PATCH_MAX)
        )

        for i,idx in enumerate(selected):

            mask=crop_patch(coord,coord[idx],PATCH_RADIUS)

            if mask.sum()<MIN_PATCH_POINTS:
                continue

            save_patch(
                coord[mask],
                normal[mask],
                label[mask],
                f"{room}_hard_{i}",
                "Area_patch_hard"
            )

            total+=1

    return total


# ===============================
# main
# ===============================

def main():

    global SOURCE_PATH
    global OUTPUT_ROOT_PATH

    SOURCE_PATH=Path(SOURCE_AREA)
    OUTPUT_ROOT_PATH=Path(OUTPUT_ROOT)

    rooms=[
        r for r in os.listdir(SOURCE_AREA)
        if os.path.isdir(os.path.join(SOURCE_AREA,r))
    ]

    print("rooms:",len(rooms))

    random.shuffle(rooms)

    train_num=int(len(rooms)*TRAIN_RATIO)

    train_rooms=rooms[:train_num]
    val_rooms=rooms[train_num:]


    # 将源房间合并为单文件 data.npy 写入输出（多线程 + 进度条）
    print("copy dataset (merge to data.npy)")

    def copy_room_merged(room, area_name):
        src = SOURCE_PATH / room
        dst_dir = OUTPUT_ROOT_PATH / area_name / room
        coord, normal, label = load_room(src)
        merge_and_save(coord, normal, label, dst_dir / "data.npy")

    all_copy_tasks = (
        [(r, "Area_1") for r in train_rooms] + [(r, "Area_5") for r in val_rooms]
    )
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        list(tqdm(
            ex.map(lambda x: copy_room_merged(x[0], x[1]), all_copy_tasks),
            total=len(all_copy_tasks),
            desc="copy",
        ))


    # patch generation（多线程 + 进度条）
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


if __name__=="__main__":

    main()