#!/usr/bin/env python3
"""
femur_stl → S3DIS 格式（v1）

一次运行同时输出增强后的 5/4/2 分类，在 output_root 下创建 c5、c4、c2 子目录：
- c5: 5 分类 0,1,2,3,4
- c4: 4 分类 1,2,3,4→0,1,2,3，原 class0 的点全部剔除不写入
- c2: 2 分类 0 vs 合并(1,2,3,4)→1

对少数类 1,3,4 做 oversample+抖动后，同一份增强点云按三种方式 remap 标签分别写入。
类别占比（原始）: 0=46.2%, 1=8.7%, 2=36.3%, 3=7.0%, 4=1.8%
"""

import os
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

# 少数类 oversample 配置（原始类别 1,3,4 占比低，按稀有程度给倍数与抖动）
# 格式: 原始类别 -> (重复倍数, 切向扰动std, 法线方向std比例)
# 法线方向 std = 切向std * 法线比例，法线比例<1 可避免沿法线“鼓包”过大
OVERSAMPLE_CONFIG = {
    1: (1, 0.15, 0.35),   # 8.7%，法线方向约为切向的 35%
    3: (4, 0.25, 0.35),
    4: (7, 0.30, 0.35),   # 1.8% 最稀有
}


def read_vertices_txt(vertices_path):
    """读 vertices.txt：X Y Z Constant → coords (N,3), labels (N,)"""
    with open(vertices_path, 'r') as f:
        lines = f.readlines()
    data_lines = []
    for line in lines[2:]:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 4:
                data_lines.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    data = np.array(data_lines)
    return data[:, :3].astype(np.float32), data[:, 3].astype(np.int32)


def load_stl_mesh(stl_path):
    """加载 STL，返回 Open3D TriangleMesh 或 None"""
    if not os.path.exists(stl_path):
        return None
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    return mesh if len(mesh.vertices) > 0 else None


def mesh_to_point_cloud(mesh, use_vertices=True):
    """网格 → 点云坐标与法向；(N,3), (N,3)"""
    if use_vertices:
        coords = np.array(mesh.vertices).astype(np.float32)
        mesh.compute_vertex_normals()
        normals = np.array(mesh.vertex_normals).astype(np.float32)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
        coords = np.array(pcd.points).astype(np.float32)
        normals = np.array(pcd.normals).astype(np.float32)
    return coords, normals


def match_labels_to_points(vertices_coords, vertices_labels, point_coords, distance_threshold=0.01, workers=-1):
    """批量半径最近邻 + 多数表决；用 scipy.cKDTree 一次查询所有点，避免逐点 Python 循环。"""
    tree = cKDTree(vertices_coords.astype(np.float64))
    radius = float(distance_threshold)
    # 一次批量半径查询（多核）
    neighbor_lists = tree.query_ball_point(point_coords.astype(np.float64), r=radius, workers=workers)
    labels = np.full(len(point_coords), -1, dtype=np.int32)
    for i, idx in enumerate(neighbor_lists):
        if len(idx) == 0:
            continue
        neighbor_labels = vertices_labels[np.asarray(idx, dtype=np.intp)]
        valid = neighbor_labels[neighbor_labels >= 0]
        if valid.size > 0:
            labels[i] = np.bincount(valid).argmax()
    return labels


def remap_segment_labels(segment, class_mode):
    """
    按 class_mode 重映射标签（c4 仅用于子集内的 remap，不在此做整表）。
    - "5": 不变
    - "4": 仅对 1,2,3,4 做 1→0,2→1,3→2,4→3（调用方先筛掉 class0）
    - "2": 0→0, 1,2,3,4→1
    """
    seg = segment.copy()
    if class_mode == "5":
        return seg
    if class_mode == "2":
        out = np.full_like(seg, -1, dtype=np.int32)
        out[seg == 0] = 0
        out[np.isin(seg, [1, 2, 3, 4])] = 1
        return out
    if class_mode == "4":
        out = np.empty_like(seg, dtype=np.int32)
        for orig, new in [(1, 0), (2, 1), (3, 2), (4, 3)]:
            out[seg == orig] = new
        return out
    return seg


def _jitter_along_normal_and_tangent(coords, normals, tangential_std, normal_ratio, rng):
    """
    各向异性抖动：切向用 tangential_std，法线方向用 tangential_std * normal_ratio（缩小沿法线扩散）。
    coords/normals: (N, 3)，normals 需已单位化。
    """
    n = normals.astype(np.float64)
    # 切向基：t1 = n × up，退化时换 up
    up = np.array([0, 0, 1], dtype=np.float64)
    t1 = np.cross(n, up)
    nrm_t1 = np.linalg.norm(t1, axis=1, keepdims=True)
    degenerate = nrm_t1.ravel() < 1e-8
    if np.any(degenerate):
        t1[degenerate] = np.cross(n[degenerate], np.array([1, 0, 0], dtype=np.float64))
        nrm_t1 = np.linalg.norm(t1, axis=1, keepdims=True)
    nrm_t1 = np.maximum(nrm_t1, 1e-8)
    t1 /= nrm_t1
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
    normal_std = float(tangential_std) * float(normal_ratio)
    a = rng.normal(0, normal_std, (len(coords), 1)).astype(np.float32)
    b = rng.normal(0, tangential_std, (len(coords), 1)).astype(np.float32)
    c = rng.normal(0, tangential_std, (len(coords), 1)).astype(np.float32)
    noise = a * n.astype(np.float32) + b * t1.astype(np.float32) + c * t2.astype(np.float32)
    return coords + noise


def oversample_minority(coord, normal, segment, config, rng):
    """
    对 config 中指定的原始类别做 oversample+抖动；先收集再一次性 concat。
    config 每项: (重复倍数, 切向std) 或 (重复倍数, 切向std, 法线方向std比例)。
    若为 3 元组则用法线方向缩小后的各向异性抖动。
    """
    extra_coords, extra_normals, extra_segments = [], [], []
    for orig_cls, val in config.items():
        tup = val if isinstance(val, (tuple, list)) else (val,)
        repeat = int(tup[0])
        if repeat <= 0:
            continue
        tangential_std = float(tup[1]) if len(tup) > 1 else 0.0
        normal_ratio = float(tup[2]) if len(tup) > 2 else 1.0  # 1.0 = 与切向同，即各向同性
        mask = segment == orig_cls
        if mask.sum() == 0:
            continue
        c = np.repeat(coord[mask], repeat, axis=0)
        nrm = np.repeat(normal[mask], repeat, axis=0)
        lbl = np.repeat(segment[mask], repeat, axis=0)
        if tangential_std > 0:
            c = _jitter_along_normal_and_tangent(c, nrm, tangential_std, normal_ratio, rng)
        extra_coords.append(c)
        extra_normals.append(nrm)
        extra_segments.append(lbl)
    if not extra_coords:
        return coord, normal, segment
    coord = np.concatenate([coord] + extra_coords, axis=0)
    normal = np.concatenate([normal] + extra_normals, axis=0)
    segment = np.concatenate([segment] + extra_segments, axis=0)
    return coord, normal, segment


# 一次运行输出三套分类的子目录名
CLASS_SUBDIRS = ("c5", "c4", "c2")


def process_patient_folder(patient_folder, output_root, area_name, oversample_config,
                           distance_threshold=0.01, include_tibia=False, seed=42, match_workers=-1):
    """
    处理单例：读 STL+vertices.txt → 匹配标签 → 少数类增强（一次）→ 按 c5/c4/c2 分别 remap 并写入。
    输出目录: output_root/c5/area/scene, output_root/c4/..., output_root/c2/...
    """
    patient_folder = Path(patient_folder)
    patient_name = patient_folder.name
    vertices_txt = patient_folder / "vertices.txt"
    femur_stl = patient_folder / "femur.stl"
    tibia_stl = patient_folder / "tibia_fibula.stl"

    if not vertices_txt.exists():
        return False
    try:
        vertices_coords, vertices_labels = read_vertices_txt(vertices_txt)
    except Exception:
        return False

    stl_list = []
    if femur_stl.exists():
        stl_list.append(femur_stl)
    if include_tibia and tibia_stl.exists():
        stl_list.append(tibia_stl)
    if not stl_list:
        return False

    all_coords, all_normals = [], []
    for stl_path in stl_list:
        mesh = load_stl_mesh(stl_path)
        if mesh is None:
            continue
        coords, normals = mesh_to_point_cloud(mesh, use_vertices=True)
        if len(coords) > 0:
            all_coords.append(coords)
            all_normals.append(normals)
    if not all_coords:
        return False

    coord = np.concatenate(all_coords, axis=0)
    normal = np.concatenate(all_normals, axis=0)

    if len(coord) == len(vertices_coords) and np.max(np.abs(coord - vertices_coords)) < distance_threshold:
        segment = vertices_labels.copy()
    else:
        segment = match_labels_to_points(
            vertices_coords, vertices_labels, coord, distance_threshold, workers=match_workers
        )

    if np.all(segment <= 0):
        segment[:] = 0

    # 对原始少数类 1,3,4 做一次 oversample，三种分类共用同一份增强点云
    rng = np.random.default_rng(seed)
    coord, normal, segment_aug = oversample_minority(coord, normal, segment.copy(), oversample_config, rng)

    # 只做一次类型转换，三套输出共用
    coord_f = coord.astype(np.float32)
    normal_f = normal.astype(np.float32)
    color = np.full((len(coord), 3), 128, dtype=np.uint8)
    scene_name = patient_name.replace(" ", "_").replace(".", "_")

    for subdir, class_mode in zip(CLASS_SUBDIRS, ("5", "4", "2")):
        output_dir = Path(output_root) / subdir / area_name / scene_name
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if class_mode == "4":
                # c4：只保留原 1,2,3,4 的点，去掉 class0，再 remap 为 0,1,2,3
                mask = np.isin(segment_aug, [1, 2, 3, 4])
                seg = remap_segment_labels(segment_aug[mask], "4")
                np.save(output_dir / "coord.npy", coord_f[mask])
                np.save(output_dir / "segment.npy", seg.astype(np.int16))
                np.save(output_dir / "normal.npy", normal_f[mask])
                np.save(output_dir / "color.npy", color[mask])
            else:
                seg = remap_segment_labels(segment_aug, class_mode)
                np.save(output_dir / "coord.npy", coord_f)
                np.save(output_dir / "segment.npy", seg.astype(np.int16))
                np.save(output_dir / "normal.npy", normal_f)
                np.save(output_dir / "color.npy", color)
        except Exception:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='femur_stl → S3DIS（v1，一次输出 c5/c4/c2 三套分类）')
    parser.add_argument('--input_root', type=str, required=True, help='输入根目录（患者子文件夹含 femur.stl + vertices.txt）')
    parser.add_argument('--output_root', type=str, required=True, help='输出根目录，其下将创建 c5、c4、c2 子目录')
    parser.add_argument('--area_name', type=str, default='Area_1', help='区域名，如 Area_1')
    parser.add_argument('--distance_threshold', type=float, default=0.01, help='标签匹配半径')
    parser.add_argument('--include_tibia', action='store_true', help='是否包含 tibia_fibula.stl')
    parser.add_argument('--patient_id', type=str, default=None, help='仅处理指定患者文件夹名')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（增强可复现）')
    parser.add_argument('--num_workers', type=int, default=12, help='处理患者并行线程数')
    parser.add_argument('--match_workers', type=int, default=-1,
                        help='标签匹配时 scipy KDTree 使用的线程数，-1 表示用满 CPU')
    # 可选：覆盖少数类增强倍数与抖动，例如 --oversample "1:2,0.15" "3:5,0.25" "4:12,0.3"
    parser.add_argument('--oversample', type=str, nargs='*', default=None,
                        help='覆盖增强配置，每项 "原始类:倍数,抖动标准差" 如 "4:12,0.3"')
    parser.add_argument('--train_test_split', action='store_true', help='按 10:1 划分训练(Area_1)/测试(Area_5)')
    parser.add_argument('--train_ratio', type=float, default=10.0/11.0, help='训练集比例')
    args = parser.parse_args()

    # 解析 oversample 配置，每项 "类:倍数,切向std[,法线比例]" 如 "4:7,0.3,0.35"
    oversample_config = dict(OVERSAMPLE_CONFIG)
    if args.oversample:
        for s in args.oversample:
            part = s.split(":")
            if len(part) != 2:
                continue
            cls = int(part[0].strip())
            rest = [x.strip() for x in part[1].strip().split(",")]
            if len(rest) >= 2:
                t = (int(rest[0]), float(rest[1]))
                if len(rest) >= 3:
                    t = t + (float(rest[2]),)
                oversample_config[cls] = t

    input_root = Path(args.input_root)
    if not input_root.exists():
        print(f"错误: 输入目录不存在: {input_root}")
        return
    patient_folders = [f for f in input_root.iterdir() if f.is_dir()]
    if args.patient_id:
        patient_folders = [f for f in patient_folders if f.name == args.patient_id]
    if not patient_folders:
        print("未找到患者文件夹")
        return

    if args.train_test_split:
        rng = np.random.default_rng(args.seed)
        idx = np.random.permutation(len(patient_folders))
        patient_folders = [patient_folders[i] for i in idx]
        num_train = int(len(patient_folders) * args.train_ratio)
        area_assignments = {pf: 'Area_1' for pf in patient_folders[:num_train]}
        for pf in patient_folders[num_train:]:
            area_assignments[pf] = 'Area_5'
        print(f"训练(Area_1): {num_train}, 测试(Area_5): {len(patient_folders) - num_train}")
    else:
        area_assignments = {pf: args.area_name for pf in patient_folders}

    success, failed = 0, []

    def run_one(pf):
        area = area_assignments[pf]
        s = args.seed + hash(pf.name) % (2 ** 20)
        ok = process_patient_folder(
            pf, args.output_root, area, oversample_config,
            args.distance_threshold, args.include_tibia, seed=s, match_workers=args.match_workers
        )
        return ok, pf.name

    if args.num_workers <= 1:
        for pf in tqdm(patient_folders, desc="处理"):
            ok, name = run_one(pf)
            if ok:
                success += 1
            else:
                failed.append(name)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(run_one, pf): pf for pf in patient_folders}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="处理"):
                ok, name = fut.result()
                if ok:
                    success += 1
                else:
                    failed.append(name)

    print(f"完成: {success}/{len(patient_folders)} 成功")
    if failed:
        print("失败:", failed[:10])
    if args.train_test_split or len(set(area_assignments.values())) > 1:
        area_counts = Counter(area_assignments.values())
        for area_name in sorted(area_counts.keys()):
            print(f"  {area_name}: {area_counts[area_name]} 例")


if __name__ == '__main__':
    main()
