#!/usr/bin/env python3
"""
对清理后的 STL / vertices 做两类可选处理：
1) 绕 Z 轴对齐（统一 XY 视角朝向）
2) 网格平滑（减弱坑洼和边缘毛刺）

设计目标：
- 标签值不变：vertices*.txt 的第 4 列 label 原样保留。
- 批量处理：按 case 子目录遍历。
- 与现有目录兼容：默认读 femur_largest.stl + vertices_largest.txt。

注意：
- `smooth_mesh` 主要改善网格视觉与几何连续性；若你希望 vertices 点位也跟着“贴合平滑后网格”，
  请同时开启 `--project_vertices_to_mesh 1`。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import open3d as o3d


def iter_cases(root: Path) -> Iterable[Path]:
    for p in root.iterdir():
        if p.is_dir():
            yield p


def load_case_strength_map(path: str) -> dict[str, int]:
    """
    读取平滑强度文件：
    每行格式：<case_name> <strength>
    例如：1772184_ZHANG_TAO 4
    """
    p = Path(path)
    out: dict[str, int] = {}
    if not str(path).strip():
        return out
    if not p.exists():
        print(f"[WARN] smooth_strength_file not found: {p}")
        return out
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip().split()
        if len(s) < 2:
            continue
        name = s[0].strip()
        try:
            v = int(float(s[1]))
        except Exception:
            continue
        if name and v > 0:
            out[name] = v
    return out


def rotation_z(angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_z_angle_by_pca_xy(points: np.ndarray) -> float:
    """
    用 XY 平面的 PCA 主轴估计角度（度）。
    返回“主轴相对 +X 方向”的角度。
    """
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return 0.0
    xy = points[:, :2].astype(np.float64, copy=False)
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = (xy.T @ xy) / max(1, len(xy) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def normalize_angle_deg(angle_deg: float) -> float:
    x = float(angle_deg)
    while x <= -180.0:
        x += 360.0
    while x > 180.0:
        x -= 360.0
    return x


def _safe_label_to_int(label_token: str) -> int | None:
    try:
        v = float(label_token)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return int(v)


def estimate_z_angle_by_semantic(
    points: np.ndarray,
    labels_str: List[str],
    class_a: int = 0,
    class_b: int = 2,
    class_valley_main: int = 0,
    class_valley_other: int = 4,
    valley_knn: int = 16,
) -> float:
    """
    语义对齐：
    - class_a / class_b：凹字两侧“突起”类（默认 0 和 2）
    - class_valley_main / class_valley_other：凹陷交接类（默认 0 和 4）

    目标：
    - 让两侧突起连线对齐到 +X 轴
    - 并确定符号，使 class0/class4 交接凹陷位于该连线“下方”（-Y）
    """
    if len(points) < 10 or len(points) != len(labels_str):
        return 0.0

    labels = np.full((len(labels_str),), -10_000, dtype=np.int32)
    for i, t in enumerate(labels_str):
        li = _safe_label_to_int(t)
        if li is not None:
            labels[i] = li

    pa = points[labels == int(class_a)]
    pb = points[labels == int(class_b)]
    p0 = points[labels == int(class_valley_main)]
    p4 = points[labels == int(class_valley_other)]
    if len(pa) < 10 or len(pb) < 10 or len(p0) < 10 or len(p4) < 10:
        # 数据不足时退回 PCA
        return -estimate_z_angle_by_pca_xy(points)

    ca = pa[:, :2].mean(axis=0)
    cb = pb[:, :2].mean(axis=0)
    v = cb - ca
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return -estimate_z_angle_by_pca_xy(points)
    v = v / nv

    # 估计 class0/class4 交接区域中心：从 class0 中挑与 class4 近邻的点
    # 不依赖 scipy，使用分块计算最小距离
    k = max(1, int(valley_knn))
    block = 4000
    dmin = np.full((len(p0),), np.inf, dtype=np.float64)
    p0xy = p0[:, :2].astype(np.float64, copy=False)
    p4xy = p4[:, :2].astype(np.float64, copy=False)
    for s in range(0, len(p0xy), block):
        e = min(len(p0xy), s + block)
        a = p0xy[s:e]  # (m,2)
        # (m,n,2) 内存较大，分块控制
        diff = a[:, None, :] - p4xy[None, :, :]
        dsq = np.sum(diff * diff, axis=2)
        dmin[s:e] = np.sqrt(np.min(dsq, axis=1))
    idx = np.argsort(dmin)[: min(k, len(dmin))]
    valley_center = p0xy[idx].mean(axis=0)

    # 先把突起连线旋到 +X
    theta = float(np.degrees(np.arctan2(v[1], v[0])))
    base_angle = -theta

    # 再确定方向符号：凹陷应在连线下方（-Y）
    # 旋转后 midpoint->valley 的 y 如果为正，则整体再翻转 180 度。
    mid = 0.5 * (ca + cb)
    R = rotation_z(base_angle)[:2, :2]
    vec_mv = (valley_center - mid) @ R.T
    if vec_mv[1] > 0:
        base_angle += 180.0

    return normalize_angle_deg(base_angle)


def estimate_z_angle_make_classes_level(
    points: np.ndarray,
    labels_str: List[str],
    class_a: int = 0,
    class_b: int = 2,
    tip_ratio: float = 0.08,
) -> float:
    """
    旋转目标：让 class_a 与 class_b 的“中心连线”水平（Δy -> 0）。
    这等价于把两类中心连线对齐到 X 轴，不额外规定左右或上下朝向。
    """
    if len(points) < 10 or len(points) != len(labels_str):
        return -estimate_z_angle_by_pca_xy(points)

    labels = np.full((len(labels_str),), -10_000, dtype=np.int32)
    for i, t in enumerate(labels_str):
        li = _safe_label_to_int(t)
        if li is not None:
            labels[i] = li

    pa = points[labels == int(class_a)]
    pb = points[labels == int(class_b)]
    if len(pa) < 10 or len(pb) < 10:
        return -estimate_z_angle_by_pca_xy(points)

    # 使用“突起端点”而不是全类质心，避免被大面积分布拉偏角度
    g = points[:, :2].mean(axis=0)
    pa_xy = pa[:, :2]
    pb_xy = pb[:, :2]
    da = np.linalg.norm(pa_xy - g[None, :], axis=1)
    db = np.linalg.norm(pb_xy - g[None, :], axis=1)
    r = float(np.clip(tip_ratio, 0.01, 0.5))
    ka = max(20, int(len(pa_xy) * r))
    kb = max(20, int(len(pb_xy) * r))
    ia = np.argsort(da)[-ka:]
    ib = np.argsort(db)[-kb:]
    ca = pa_xy[ia].mean(axis=0)
    cb = pb_xy[ib].mean(axis=0)
    v = cb - ca
    if float(np.linalg.norm(v)) < 1e-8:
        return -estimate_z_angle_by_pca_xy(points)

    theta = float(np.degrees(np.arctan2(v[1], v[0])))
    # 旋转 -theta 后，连线与 X 轴平行 => 两类在 y 上同高
    return normalize_angle_deg(-theta)


def enforce_class_left_rule(
    points: np.ndarray,
    labels_str: List[str],
    class_left: int = 0,
    class_right: int = 2,
    min_points: int = 10,
) -> bool:
    """
    判断是否满足“class_left 在 class_right 左侧”。
    返回 True 表示已满足；False 表示应再旋转 180°。
    """
    if len(points) < min_points or len(points) != len(labels_str):
        return True
    labels = np.full((len(labels_str),), -10_000, dtype=np.int32)
    for i, t in enumerate(labels_str):
        li = _safe_label_to_int(t)
        if li is not None:
            labels[i] = li
    pl = points[labels == int(class_left)]
    pr = points[labels == int(class_right)]
    if len(pl) < min_points or len(pr) < min_points:
        return True
    xl = float(np.mean(pl[:, 0]))
    xr = float(np.mean(pr[:, 0]))
    return xl < xr


def should_flip_along_x_axis_by_classes(
    points: np.ndarray,
    labels_str: List[str],
    class_right: int = 0,
    class_left: int = 2,
    class_upper: int = 4,
    min_points: int = 10,
) -> bool:
    """
    条件触发（True 则执行沿 x 轴镜像）：
    - class_right 在右侧（x 更大）
    - class_left 在左侧（x 更小）
    - class_upper 在两者上方（y 更小）
    """
    if len(points) < min_points or len(points) != len(labels_str):
        return False
    labels = np.full((len(labels_str),), -10_000, dtype=np.int32)
    for i, t in enumerate(labels_str):
        li = _safe_label_to_int(t)
        if li is not None:
            labels[i] = li
    pr = points[labels == int(class_right)]
    pl = points[labels == int(class_left)]
    pu = points[labels == int(class_upper)]
    if len(pr) < min_points or len(pl) < min_points or len(pu) < min_points:
        return False

    xr = float(np.mean(pr[:, 0]))
    xl = float(np.mean(pl[:, 0]))
    yr = float(np.mean(pr[:, 1]))
    yl = float(np.mean(pl[:, 1]))
    yu = float(np.mean(pu[:, 1]))
    return (xr > xl) and (yu < min(yr, yl))


def mirror_along_zy_plane(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    沿 zy 平面做镜像（关于 center_x 对称）：x 镜像，y/z 不变。
    """
    out = np.array(points, dtype=np.float64, copy=True)
    out[:, 0] = 2.0 * float(center[0, 0]) - out[:, 0]
    return out


def load_vertices_txt_lines(path: Path) -> Tuple[str, List[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"bad vertices txt: {path}")
    header = lines[0]
    data_lines = lines[2:]  # 跳过注释行和数量行
    return header, data_lines


def parse_vertices_xyz_label(data_lines: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    返回：
    - xyz: (N,3) float64
    - labels_str: 长度 N，原始 label token 字符串（原样保留）
    - row_ids: 对应 data_lines 的行号（用于回写）
    """
    xyz = []
    labels = []
    row_ids = []
    for i, ln in enumerate(data_lines):
        s = ln.strip().split()
        if len(s) < 4:
            continue
        try:
            x = float(s[0])
            y = float(s[1])
            z = float(s[2])
        except Exception:
            continue
        xyz.append([x, y, z])
        labels.append(s[3])  # label 字符串原样保留
        row_ids.append(i)
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float64), [], np.zeros((0,), dtype=np.int64)
    return np.asarray(xyz, dtype=np.float64), labels, np.asarray(row_ids, dtype=np.int64)


def apply_transform(points: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    return (points - center) @ R.T + center


def smooth_mesh(mesh: o3d.geometry.TriangleMesh, method: str, iters: int, lam: float, mu: float) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.remove_non_manifold_edges()

    if method == "taubin":
        m = m.filter_smooth_taubin(number_of_iterations=max(1, int(iters)), lambda_filter=float(lam), mu=float(mu))
    else:
        m = m.filter_smooth_laplacian(number_of_iterations=max(1, int(iters)), lambda_filter=float(lam))
    m.compute_vertex_normals()
    return m


def project_points_to_mesh(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    if len(points) == 0:
        return points
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    p = o3d.core.Tensor(points.astype(np.float32))
    ans = scene.compute_closest_points(p)
    return np.asarray(ans["points"].numpy(), dtype=np.float64)


def write_vertices_txt(
    out_path: Path,
    header: str,
    data_lines: List[str],
    row_ids: np.ndarray,
    xyz_new: np.ndarray,
    labels_str: List[str],
) -> None:
    # 只替换有效行；无效行维持原样
    out_lines = list(data_lines)
    for k, rid in enumerate(row_ids.tolist()):
        x, y, z = xyz_new[k]
        out_lines[rid] = f"{x:.8f} {y:.8f} {z:.8f} {labels_str[k]}"

    final = [header, str(int(len(row_ids)))] + out_lines
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(final) + "\n", encoding="utf-8")


def process_case(
    case_dir: Path,
    out_root: Path,
    stl_name: str,
    vertices_name: str,
    stl_out_name: str,
    vertices_out_name: str,
    align_z: bool,
    align_mode: str,
    z_angle_deg: float,
    semantic_class_a: int,
    semantic_class_b: int,
    semantic_valley_main: int,
    semantic_valley_other: int,
    semantic_valley_knn: int,
    level_tip_ratio: float,
    auto_mirror_zy_rule: bool,
    mirror_class_right: int,
    mirror_class_left: int,
    mirror_class_upper: int,
    force_mirror_zy_cases: set[str],
    extra_rot180_cases: set[str],
    extra_flip_zy_after_rot180: bool,
    smooth_strength_map: dict[str, int],
    smooth_strength_mode: str,
    smooth_mesh_flag: bool,
    smooth_method: str,
    smooth_iters: int,
    smooth_lambda: float,
    smooth_mu: float,
    project_vertices_to_mesh: bool,
) -> None:
    stl_path = case_dir / stl_name
    vtxt_path = case_dir / vertices_name
    if not stl_path.exists() or not vtxt_path.exists():
        print(f"skip: {case_dir.name}")
        return

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print(f"skip empty mesh: {case_dir.name}")
        return

    header, data_lines = load_vertices_txt_lines(vtxt_path)
    xyz, labels, row_ids = parse_vertices_xyz_label(data_lines)
    if len(xyz) == 0:
        print(f"skip empty vertices: {case_dir.name}")
        return

    center = xyz.mean(axis=0, keepdims=True)
    mesh_v = np.asarray(mesh.vertices).astype(np.float64)
    mesh_center = mesh_v.mean(axis=0, keepdims=True)

    # 1) 对齐：默认用 vertices 的 PCA 方向
    use_angle = 0.0
    if align_z:
        if align_mode == "manual":
            use_angle = float(z_angle_deg)
        elif align_mode == "level02":
            use_angle = estimate_z_angle_make_classes_level(
                xyz,
                labels,
                class_a=int(semantic_class_a),
                class_b=int(semantic_class_b),
                tip_ratio=float(level_tip_ratio),
            )
        elif align_mode == "semantic":
            use_angle = estimate_z_angle_by_semantic(
                xyz,
                labels,
                class_a=int(semantic_class_a),
                class_b=int(semantic_class_b),
                class_valley_main=int(semantic_valley_main),
                class_valley_other=int(semantic_valley_other),
                valley_knn=int(semantic_valley_knn),
            )
        else:
            est = estimate_z_angle_by_pca_xy(xyz)
            # 将主轴旋到 +X 方向 => 旋转 -est
            use_angle = -est
        use_angle = normalize_angle_deg(use_angle)
        R = rotation_z(use_angle)
        xyz = apply_transform(xyz, R, center=center)
        mesh_v = apply_transform(mesh_v, R, center=mesh_center)
        mesh.vertices = o3d.utility.Vector3dVector(mesh_v)

        # 先按规则/名单做 zy 平面镜像（x 翻转）：避免 180° 旋转把上下关系也带偏
        need_mirror_zy = False
        if auto_mirror_zy_rule and case_dir.name in force_mirror_zy_cases:
            need_mirror_zy = True
        elif auto_mirror_zy_rule and should_flip_along_x_axis_by_classes(
            xyz,
            labels,
            class_right=int(mirror_class_right),
            class_left=int(mirror_class_left),
            class_upper=int(mirror_class_upper),
        ):
            need_mirror_zy = True

        if need_mirror_zy:
            xyz = mirror_along_zy_plane(xyz, center=center)
            mesh_v = mirror_along_zy_plane(np.asarray(mesh.vertices).astype(np.float64), center=mesh_center)
            mesh.vertices = o3d.utility.Vector3dVector(mesh_v)

        # 特殊名单：额外绕 z 轴旋转 180 度
        if case_dir.name in extra_rot180_cases:
            Rf = rotation_z(180.0)
            xyz = apply_transform(xyz, Rf, center=center)
            mesh_v = apply_transform(np.asarray(mesh.vertices).astype(np.float64), Rf, center=mesh_center)
            mesh.vertices = o3d.utility.Vector3dVector(mesh_v)
            use_angle = normalize_angle_deg(use_angle + 180.0)
            if extra_flip_zy_after_rot180:
                xyz = mirror_along_zy_plane(xyz, center=center)
                mesh_v = mirror_along_zy_plane(np.asarray(mesh.vertices).astype(np.float64), center=mesh_center)
                mesh.vertices = o3d.utility.Vector3dVector(mesh_v)

    # 2) 平滑
    if smooth_mesh_flag:
        iters_use = int(smooth_iters)
        if case_dir.name in smooth_strength_map:
            s = int(smooth_strength_map[case_dir.name])
            if smooth_strength_mode == "multiply":
                iters_use = max(1, int(round(iters_use * s)))
            else:
                # as_iters: 文件值直接作为迭代次数
                iters_use = max(1, s)
        mesh = smooth_mesh(mesh, method=smooth_method, iters=iters_use, lam=smooth_lambda, mu=smooth_mu)
        if project_vertices_to_mesh:
            xyz = project_points_to_mesh(xyz, mesh)

    mesh.compute_vertex_normals()

    out_case = out_root / case_dir.name
    out_case.mkdir(parents=True, exist_ok=True)
    out_stl = out_case / stl_out_name
    out_vtxt = out_case / vertices_out_name

    ok = o3d.io.write_triangle_mesh(str(out_stl), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"failed to write stl: {out_stl}")

    write_vertices_txt(out_vtxt, header, data_lines, row_ids, xyz, labels)
    print(
        f"[OK] {case_dir.name} | angle={use_angle:.3f} deg | "
        f"smooth={int(smooth_mesh_flag)} | projected={int(project_vertices_to_mesh)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="输入目录（含各 case 子目录）")
    ap.add_argument("--out_root", required=True, help="输出目录（保留 case 子目录结构）")

    ap.add_argument("--stl_name", default="femur_largest.stl")
    ap.add_argument("--vertices_name", default="vertices_largest.txt")
    ap.add_argument("--stl_out_name", default="femur_largest_aligned.stl")
    ap.add_argument("--vertices_out_name", default="vertices_largest_aligned.txt")

    # 开关1：Z轴对齐
    ap.add_argument("--align_z", type=int, default=1, help="是否绕 Z 轴旋转对齐（1/0）")
    ap.add_argument(
        "--align_mode",
        choices=["pca", "manual", "semantic", "level02"],
        default="level02",
        help="pca=主轴对齐，manual=手动角度，semantic=突起+凹陷约束，level02=强制 class0/class2 在 Y 上同高",
    )
    ap.add_argument("--z_angle_deg", type=float, default=0.0, help="手动模式下绕 Z 轴旋转角度（度）")
    ap.add_argument("--semantic_class_a", type=int, default=0, help="语义对齐：突起 A 类别（默认 0）")
    ap.add_argument("--semantic_class_b", type=int, default=2, help="语义对齐：突起 B 类别（默认 2）")
    ap.add_argument("--semantic_valley_main", type=int, default=0, help="语义对齐：凹陷交接主类（默认 0）")
    ap.add_argument("--semantic_valley_other", type=int, default=4, help="语义对齐：凹陷交接另一类（默认 4）")
    ap.add_argument("--semantic_valley_knn", type=int, default=32, help="语义对齐：估计凹陷中心时选取最近邻点数")
    ap.add_argument(
        "--level_tip_ratio",
        type=float,
        default=0.08,
        help="level02 模式下用于估计突起端点的远端点比例（0.01~0.5）",
    )
    # NOTE: 已按需求移除 180° flip 逻辑（auto_flip_lr/enforce_class_left_rule）
    ap.add_argument("--auto_mirror_zy_rule", type=int, default=0, help="是否启用条件 zy 平面镜像规则（x 翻转）（1/0）")
    ap.add_argument("--mirror_class_right", type=int, default=0, help="zy镜像规则：右侧类别（默认 0）")
    ap.add_argument("--mirror_class_left", type=int, default=2, help="zy镜像规则：左侧类别（默认 2）")
    ap.add_argument("--mirror_class_upper", type=int, default=4, help="zy镜像规则：上方类别（默认 4，y 更小）")
    ap.add_argument(
        "--force_mirror_zy_cases",
        type=str,
        default="",
        help="强制 zy 镜像的病例名（逗号分隔），例如: 1934031_CHEN_ZHUO,2153208_HAO_LIN_YU",
    )
    ap.add_argument(
        "--extra_rot180_cases_file",
        type=str,
        default="D:/Projects/by_pointcept/data/rot.txt",
        help="每行一个病例名；命中后额外绕 z 轴旋转 180 度",
    )
    ap.add_argument(
        "--extra_flip_zy_after_rot180",
        type=int,
        default=1,
        help="对 extra_rot180_cases 命中的病例：旋转180后是否再做一次 zy 平面镜像（x翻转）（1/0）",
    )
    ap.add_argument(
        "--smooth_strength_file",
        type=str,
        default="D:/Projects/by_pointcept/data/smooth.txt",
        help="每行 `<case_name> <strength>`，按病例覆盖平滑强度",
    )
    ap.add_argument(
        "--smooth_strength_mode",
        choices=["as_iters", "multiply"],
        default="as_iters",
        help="as_iters=文件值直接作为平滑迭代次数；multiply=文件值作为默认迭代次数倍数",
    )

    # 开关2：平滑/补坑洼感
    ap.add_argument("--smooth_mesh", type=int, default=0, help="是否对网格做平滑（1/0）")
    ap.add_argument("--smooth_method", choices=["taubin", "laplacian"], default="taubin")
    ap.add_argument("--smooth_iters", type=int, default=10)
    ap.add_argument("--smooth_lambda", type=float, default=0.5)
    ap.add_argument("--smooth_mu", type=float, default=-0.53, help="仅 taubin 生效")
    ap.add_argument(
        "--project_vertices_to_mesh",
        type=int,
        default=0,
        help="平滑后是否将 vertices 点投影到网格表面（1/0）",
    )

    args = ap.parse_args()
    in_root = Path(args.input_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = list(iter_cases(in_root))
    force_mirror_zy_cases = {
        x.strip() for x in str(args.force_mirror_zy_cases).split(",") if x.strip()
    }
    extra_rot180_cases: set[str] = set()
    rot_file = str(args.extra_rot180_cases_file).strip()
    if rot_file:
        p = Path(rot_file)
        if p.exists():
            extra_rot180_cases = {
                ln.strip()
                for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()
                if ln.strip()
            }
        else:
            print(f"[WARN] extra_rot180_cases_file not found: {p}")
    smooth_strength_map = load_case_strength_map(str(args.smooth_strength_file))
    print(f"Total cases: {len(cases)}")
    for case in cases:
        try:
            process_case(
                case_dir=case,
                out_root=out_root,
                stl_name=str(args.stl_name),
                vertices_name=str(args.vertices_name),
                stl_out_name=str(args.stl_out_name),
                vertices_out_name=str(args.vertices_out_name),
                align_z=bool(args.align_z),
                align_mode=str(args.align_mode),
                z_angle_deg=float(args.z_angle_deg),
                semantic_class_a=int(args.semantic_class_a),
                semantic_class_b=int(args.semantic_class_b),
                semantic_valley_main=int(args.semantic_valley_main),
                semantic_valley_other=int(args.semantic_valley_other),
                semantic_valley_knn=int(args.semantic_valley_knn),
                level_tip_ratio=float(args.level_tip_ratio),
                auto_mirror_zy_rule=bool(args.auto_mirror_zy_rule),
                mirror_class_right=int(args.mirror_class_right),
                mirror_class_left=int(args.mirror_class_left),
                mirror_class_upper=int(args.mirror_class_upper),
                force_mirror_zy_cases=force_mirror_zy_cases,
                extra_rot180_cases=extra_rot180_cases,
                extra_flip_zy_after_rot180=bool(args.extra_flip_zy_after_rot180),
                smooth_strength_map=smooth_strength_map,
                smooth_strength_mode=str(args.smooth_strength_mode),
                smooth_mesh_flag=bool(args.smooth_mesh),
                smooth_method=str(args.smooth_method),
                smooth_iters=int(args.smooth_iters),
                smooth_lambda=float(args.smooth_lambda),
                smooth_mu=float(args.smooth_mu),
                project_vertices_to_mesh=bool(args.project_vertices_to_mesh),
            )
        except Exception as e:
            print(f"[FAIL] {case.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

