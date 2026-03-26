#!/usr/bin/env python3
"""
将“清理后的 STL（最大连通块）+ 对应 vertices_largest.txt”转换为 S3DIS 风格 npy，并保存三视图。

输入目录（input_root）结构示例：
  input_root/
    1009141_WU_JIA_JUN/
      femur_largest.stl
      vertices_largest.txt
      femur_largest.kept_vertex_indices.txt

输出目录（output_root）结构与 convert_v3.py 保持一致：
  output_root/
    c5/Area_1/<case_name>/
      coord.npy
      normal.npy
      segment.npy
      triview.png (可选)
      triview_boundary34.png (可选)

关键约束：
- x y z 与数值 label 原样使用（不换轴）；不做 KDTree/nearest neighbor 匹配。
- 注意：convert_v3.py 的 read_vertices_txt 会跳过第 4 列为 NaN 的行；若顶点文件里大量 NaN 标签，
  默认会“点变少”。可用 --vertices_nan_policy zero 将 NaN 标签映射为 0 以保留这些点。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import importlib.util
import numpy as np
import open3d as o3d
from tqdm import tqdm

def _load_convert_v3_module() -> object:
    """
    通过文件路径动态加载 convert_v3.py。

    原因：目录名 `01_data_process` 不是合法 Python 包名，无法用 `from by_scripts.01_data_process...` 正常 import。
    """
    # 本文件：by_scripts/01_data_process/clear_stls/convert_cleaned_to_s3dis_v3.py
    # 目标：  by_scripts/01_data_process/stl_to_s3dis/convert_v3.py
    convert_v3_path = Path(__file__).resolve().parents[1] / "stl_to_s3dis" / "convert_v3.py"
    if not convert_v3_path.exists():
        raise FileNotFoundError(convert_v3_path)

    spec = importlib.util.spec_from_file_location("convert_v3_dynamic", str(convert_v3_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create import spec for convert_v3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_CV3 = _load_convert_v3_module()
remap = _CV3.remap
sanity_check = _CV3.sanity_check
save_triview_images = _CV3.save_triview_images
save_boundary34_triview_images = _CV3.save_boundary34_triview_images


def read_vertices_txt_nan_policy(path: Path, nan_policy: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 convert_v3.read_vertices_txt 相同格式；对 Constant 列的 NaN 处理：
    - skip：与 convert_v3 一致，跳过该行
    - zero：NaN 视为标签 0，保留该点坐标
    """
    coord = []
    label = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for line in lines[2:]:
        s = line.strip().split()
        if len(s) < 4:
            continue
        try:
            x = float(s[0])
            y = float(s[1])
            z = float(s[2])
            l = float(s[3])
            if np.isnan(l):
                if nan_policy == "skip":
                    continue
                if nan_policy == "zero":
                    l = 0.0
                else:
                    continue
            coord.append([x, y, z])
            label.append(int(l))
        except Exception:
            continue
    return np.array(coord, dtype=np.float32), np.array(label, dtype=np.int32)


def _read_kept_indices(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    ids = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ids.append(int(s))
            except Exception:
                continue
    if not ids:
        raise RuntimeError(f"empty kept indices: {path}")
    arr = np.array(ids, dtype=np.int64)
    # 这里不强制排序/unique：保持与 vertices_largest.txt 的写出顺序一致（默认就是 sorted unique）
    return arr


def process_case_cleaned(
    case_dir: Path,
    output_root: Path,
    stl_name: str,
    vertices_name: str,
    kept_vertex_indices_name: str,
    normal_mode: str,
    normal_knn: int,
    viz: bool,
    viz_max_points: int,
    viz_point_size: float,
    viz_boundary34: bool,
    viz_boundary34_radius: float,
    viz_boundary34_max_points_det: int,
    viz_boundary34_point_size: float,
    vertices_nan_policy: str,
) -> None:
    case_name = case_dir.name

    stl_path = case_dir / stl_name
    vertices_path = case_dir / vertices_name
    kept_idx_path = case_dir / kept_vertex_indices_name

    # 兼容 aligned 输出命名
    if not stl_path.exists():
        alt_stl = case_dir / "femur_largest_aligned.stl"
        if alt_stl.exists():
            stl_path = alt_stl
    if not vertices_path.exists():
        alt_v = case_dir / "vertices_largest_aligned.txt"
        if alt_v.exists():
            vertices_path = alt_v

    if not stl_path.exists() or not vertices_path.exists():
        print("skip:", case_name)
        return

    v_coord, v_label = read_vertices_txt_nan_policy(vertices_path, nan_policy=vertices_nan_policy)
    kept_idx = _read_kept_indices(kept_idx_path) if kept_idx_path.exists() else None

    if v_coord.size == 0 or len(v_label) == 0:
        print(f"skip empty vertices_largest (0 valid rows after parse): {case_name} | {vertices_path}")
        return

    # segment 直接来自 vertices_largest.txt，保证不改 label
    segment = v_label.astype(np.int32, copy=False)
    sanity_check(segment)

    coord_out = v_coord.astype(np.float32, copy=False)

    if normal_mode == "zeros":
        normal_out = np.zeros((coord_out.shape[0], 3), dtype=np.float32)
    elif normal_mode == "mesh_kept":
        # 只有在 kept_idx 和 vertices_largest 严格同一索引空间时才可用
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            print("skip empty stl:", case_name)
            return
        mesh.compute_vertex_normals()
        normals_all = np.asarray(mesh.vertex_normals).astype(np.float32)

        if kept_idx is None:
            raise RuntimeError(f"{case_name}: normal_mode=mesh_kept but kept indices file missing")
        kept_idx = kept_idx.astype(np.int64, copy=False)
        if kept_idx.size != coord_out.shape[0]:
            raise RuntimeError(
                f"{case_name}: normal_mode=mesh_kept requires kept_idx size == vertices_largest rows "
                f"({kept_idx.size} vs {coord_out.shape[0]})"
            )
        normal_out = normals_all[kept_idx]
    else:
        # default: estimate normals directly on vertices_largest point cloud
        n_pts = int(coord_out.shape[0])
        if n_pts < 3:
            # KNN 法向在点数过少时不稳定 / Open3D 可能失败，退化为零法向
            normal_out = np.zeros((n_pts, 3), dtype=np.float32)
        else:
            knn = min(int(normal_knn), n_pts)
            knn = max(3, knn)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord_out.astype(np.float64, copy=False))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
            normal_out = np.asarray(pcd.normals).astype(np.float32)

    seg_out = remap(segment, mode="5").astype(np.int16, copy=False)

    out_dir = output_root / "c5" / "Area_1" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "coord.npy", coord_out.astype(np.float32))
    np.save(out_dir / "normal.npy", normal_out.astype(np.float32))
    np.save(out_dir / "segment.npy", seg_out.astype(np.int16))

    if viz:
        save_triview_images(
            coord_out,
            seg_out,
            save_dir=out_dir,
            max_points=viz_max_points,
            point_size=viz_point_size,
        )

    if viz_boundary34:
        save_boundary34_triview_images(
            coord_out,
            seg_out,
            save_dir=out_dir,
            boundary_radius=viz_boundary34_radius,
            max_points_det=viz_boundary34_max_points_det,
            point_size=viz_boundary34_point_size,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="清理后数据根目录（包含各 case 子目录）")
    ap.add_argument("--output_root", required=True, help="输出 npy / triview 的根目录")
    ap.add_argument("--stl_name", default="femur_largest.stl")
    ap.add_argument("--vertices_name", default="vertices_largest.txt")
    ap.add_argument("--kept_vertex_indices_name", default="femur_largest.kept_vertex_indices.txt")
    ap.add_argument(
        "--normal_mode",
        default="estimate",
        choices=["estimate", "mesh_kept", "zeros"],
        help=(
            "法向来源："
            "estimate=在 vertices_largest 点云上估计（默认，稳且不依赖 kept indices）；"
            "mesh_kept=使用 STL 的 vertex_normals 并按 kept_vertex_indices 对齐（仅当两者索引空间一致时可用）；"
            "zeros=全零法向"
        ),
    )
    ap.add_argument("--normal_knn", type=int, default=30, help="normal_mode=estimate 时的 KNN")
    ap.add_argument(
        "--vertices_nan_policy",
        default="zero",
        choices=["skip", "zero"],
        help="第 4 列 label 为 NaN 时：skip=与 convert_v3 一致丢弃该行；zero=标签记为 0 并保留点",
    )

    ap.add_argument("--viz", action="store_true", help="保存每个 case 的三视图 png")
    ap.add_argument("--viz_max_points", type=int, default=300000, help="三视图渲染最多点数")
    ap.add_argument("--viz_point_size", type=float, default=0.2, help="三视图点大小")
    ap.add_argument("--viz_boundary34", action="store_true", help="保存 class 3/4 交接处的局部三视图")
    ap.add_argument("--viz_boundary34_radius", type=float, default=0.005)
    ap.add_argument("--viz_boundary34_max_points_det", type=int, default=200000)
    ap.add_argument("--viz_boundary34_point_size", type=float, default=0.3)

    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    cases = [p for p in input_root.iterdir() if p.is_dir()]
    print("Total cases:", len(cases))

    for case in tqdm(cases):
        process_case_cleaned(
            case_dir=case,
            output_root=output_root,
            stl_name=str(args.stl_name),
            vertices_name=str(args.vertices_name),
            kept_vertex_indices_name=str(args.kept_vertex_indices_name),
            normal_mode=str(args.normal_mode),
            normal_knn=int(args.normal_knn),
            viz=bool(args.viz),
            viz_max_points=int(args.viz_max_points),
            viz_point_size=float(args.viz_point_size),
            viz_boundary34=bool(args.viz_boundary34),
            viz_boundary34_radius=float(args.viz_boundary34_radius),
            viz_boundary34_max_points_det=int(args.viz_boundary34_max_points_det),
            viz_boundary34_point_size=float(args.viz_boundary34_point_size),
            vertices_nan_policy=str(args.vertices_nan_policy),
        )


if __name__ == "__main__":
    main()

