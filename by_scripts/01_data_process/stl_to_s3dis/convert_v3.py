import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    # 使用离线渲染保存图片，避免无图形环境下 plt/open3d 弹窗失败
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------------------------------------
# read vertices.txt
# ------------------------------------------------

def read_vertices_txt(path):

    coord = []
    label = []

    with open(path) as f:
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
                continue

            coord.append([x, y, z])
            label.append(int(l))

        except:
            continue

    coord = np.array(coord, dtype=np.float32)
    label = np.array(label, dtype=np.int32)

    print("valid vertices:", len(coord))

    return coord, label


# ------------------------------------------------
# load STL
# ------------------------------------------------

def load_mesh(path):

    mesh = o3d.io.read_triangle_mesh(str(path))

    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh: {path}")

    mesh.compute_vertex_normals()

    coord = np.asarray(mesh.vertices).astype(np.float32)
    normal = np.asarray(mesh.vertex_normals).astype(np.float32)

    return coord, normal


# ------------------------------------------------
# scene info
# ------------------------------------------------

def scene_info(coord):

    bbox = coord.max(axis=0) - coord.min(axis=0)

    size = np.linalg.norm(bbox)

    print("scene bbox:", bbox)
    print("scene size:", size)

    return size


# ------------------------------------------------
# label matching (robust)
# ------------------------------------------------

def match_labels(v_coord, v_label, coord):

    tree = cKDTree(v_coord)

    dist, idx = tree.query(coord, k=1)

    labels = v_label[idx]

    return labels


# ------------------------------------------------
# sanity check
# ------------------------------------------------

def sanity_check(segment):

    labels, counts = np.unique(segment, return_counts=True)

    total = len(segment)

    print("\nLabel distribution:")

    for l, c in zip(labels, counts):

        print(f"class {l}: {c} ({c/total:.3f})")

    if len(labels) <= 1:

        print("WARNING: only one label detected")


# ------------------------------------------------
# label remap
# ------------------------------------------------

def remap(segment, mode):

    # 统一保证标签为非负整数
    segment = np.asarray(segment, dtype=np.int32)
    segment[segment < 0] = 0

    if mode == "5":

        # 5 类：标签范围 [0, 4]
        return np.clip(segment, 0, 4)

    if mode == "4":

        # 只保留大于 0 的点，将 1~4 映射为 0~3，并裁剪到 [0, 3]
        mask = segment > 0
        seg4 = segment[mask] - 1
        seg4 = np.clip(seg4, 0, 3)
        return seg4

    if mode == "2":

        # 2 类：显式约束为 0 / 1
        out = np.zeros_like(segment, dtype=np.int32)
        out[segment > 0] = 1
        return out


# ------------------------------------------------
# triview visualization (XY / YZ / XZ)
# ------------------------------------------------

_CLASS_COLORS_5 = np.array(
    [
        [0.894, 0.102, 0.110],  # class 0 - red
        [0.216, 0.494, 0.722],  # class 1 - blue
        [0.302, 0.686, 0.290],  # class 2 - green
        [0.596, 0.306, 0.639],  # class 3 - purple
        [1.000, 0.498, 0.000],  # class 4 - orange
    ],
    dtype=np.float32,
)


def save_triview_images(
    coord,
    segment,
    save_dir,
    max_points=300000,
    point_size=0.2,
    dpi=200,
):
    """
    保存三视图到 save_dir：
    - `triview.png`：一张图里包含 XY / YZ / XZ（仅输出合并图）

    三视图使用正交投影的 2D 散点图（不依赖相机渲染）。
    """
    if plt is None:
        print("WARNING: matplotlib 未安装或不可用，跳过三视图可视化")
        return

    coord = np.asarray(coord, dtype=np.float32)
    segment = np.asarray(segment, dtype=np.int32)
    assert coord.ndim == 2 and coord.shape[1] == 3
    assert segment.ndim == 1 and len(segment) == len(coord)

    n = len(coord)
    if n == 0:
        print("WARNING: empty coord, skip triview")
        return

    # 为了速度只渲染少量点
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        coord_plot = coord[idx]
        seg_plot = segment[idx]
    else:
        coord_plot = coord
        seg_plot = segment

    colors = np.full((len(seg_plot), 3), 0.5, dtype=np.float32)
    valid = (seg_plot >= 0) & (seg_plot < 5)
    colors[valid] = _CLASS_COLORS_5[seg_plot[valid]]

    x = coord_plot[:, 0]
    y = coord_plot[:, 1]
    z = coord_plot[:, 2]

    # 组合图：一张图包含三个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.scatter(x, y, c=colors, s=point_size, linewidths=0)
    ax.set_title("Triview XY")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    ax.scatter(y, z, c=colors, s=point_size, linewidths=0)
    ax.set_title("Triview YZ")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[2]
    ax.scatter(x, z, c=colors, s=point_size, linewidths=0)
    ax.set_title("Triview XZ")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    out_path = Path(save_dir) / "triview.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------
# boundary visualization (class 3 / class 4)
# ------------------------------------------------

def save_boundary34_triview_images(
    coord,
    segment,
    save_dir,
    boundary_radius=0.005,
    max_points_det=200000,
    point_size=0.3,
    dpi=250,
    pad_ratio=0.05,
):
    """
    输出 `triview_boundary34.png`：
    - 仅保留“class3/class4 且在 boundary_radius 内互为近邻”的交接点
    - XY/YZ/XZ 三个子图合并到一张图，并自动缩放到交接局部 bbox
    """
    if plt is None:
        print("WARNING: matplotlib 未安装或不可用，跳过 boundary 可视化")
        return

    coord = np.asarray(coord, dtype=np.float32)
    segment = np.asarray(segment, dtype=np.int32)
    assert coord.ndim == 2 and coord.shape[1] == 3
    assert segment.ndim == 1 and len(segment) == len(coord)

    mask34 = (segment == 3) | (segment == 4)
    if not np.any(mask34):
        print("WARNING: no class 3/4 points, skip boundary34")
        return

    # 为了速度：只在检测阶段采样一部分点，但 bbox/局部效果主要来自边界区域。
    idx34 = np.where(mask34)[0]
    if len(idx34) > max_points_det:
        idx34_det = np.random.choice(idx34, max_points_det, replace=False)
    else:
        idx34_det = idx34

    coord34_det = coord[idx34_det]
    seg34_det = segment[idx34_det]

    mask3 = seg34_det == 3
    mask4 = seg34_det == 4

    # 构建完整 class3/class4 KDTree，用于 radius 判定（更准确）
    idx3_full = np.where(segment == 3)[0]
    idx4_full = np.where(segment == 4)[0]

    if len(idx3_full) == 0 or len(idx4_full) == 0:
        print("WARNING: missing class 3 or 4, skip boundary34")
        return

    tree3 = cKDTree(coord[idx3_full])
    tree4 = cKDTree(coord[idx4_full])

    boundary_mask_det = np.zeros(len(idx34_det), dtype=bool)

    # class3 点：若最近 class4 点距离 <= boundary_radius，则属于交接
    if np.any(mask3):
        dist_to_4, _ = tree4.query(coord34_det[mask3], k=1)
        boundary_mask_det[mask3] = dist_to_4 <= boundary_radius

    # class4 点：若最近 class3 点距离 <= boundary_radius，则属于交接
    if np.any(mask4):
        dist_to_3, _ = tree3.query(coord34_det[mask4], k=1)
        boundary_mask_det[mask4] = dist_to_3 <= boundary_radius

    boundary_idx_det = idx34_det[boundary_mask_det]
    if len(boundary_idx_det) == 0:
        print("WARNING: boundary34 empty after radius filter, skip boundary34")
        return

    boundary_coord = coord[boundary_idx_det]
    boundary_seg = segment[boundary_idx_det]

    # 局部 bbox（投影到 2D 用于缩放视图）
    x = boundary_coord[:, 0]
    y = boundary_coord[:, 1]
    z = boundary_coord[:, 2]

    def _limits(a):
        lo = float(np.min(a))
        hi = float(np.max(a))
        span = hi - lo
        pad = max(span * pad_ratio, 1e-6)
        return lo - pad, hi + pad

    xlim = _limits(x)
    ylim_xy = _limits(y)
    ylim_yz = _limits(z)
    zlim_xz = _limits(z)

    # 颜色：只区分 class3 / class4
    colors = np.full((len(boundary_coord), 3), 0.5, dtype=np.float32)
    colors[boundary_seg == 3] = _CLASS_COLORS_5[3]
    colors[boundary_seg == 4] = _CLASS_COLORS_5[4]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.scatter(x, y, c=colors, s=point_size, linewidths=0)
    ax.set_title(f"Boundary34 XY (r={boundary_radius})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim_xy)

    ax = axes[1]
    ax.scatter(y, z, c=colors, s=point_size, linewidths=0)
    ax.set_title(f"Boundary34 YZ (r={boundary_radius})")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*ylim_xy)
    ax.set_ylim(*ylim_yz)

    ax = axes[2]
    ax.scatter(x, z, c=colors, s=point_size, linewidths=0)
    ax.set_title(f"Boundary34 XZ (r={boundary_radius})")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim_xz)

    plt.tight_layout()
    out_path = Path(save_dir) / "triview_boundary34.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------
# process case
# ------------------------------------------------

def process_case(
    case_dir,
    output_root,
    viz=False,
    viz_max_points=300000,
    viz_point_size=0.2,
    viz_boundary34=False,
    viz_boundary34_radius=0.005,
    viz_boundary34_max_points_det=200000,
    viz_boundary34_point_size=0.3,
):

    case_name = case_dir.name

    stl = case_dir / "femur.stl"
    vert = case_dir / "vertices.txt"

    if not stl.exists() or not vert.exists():

        print("skip:", case_name)
        return


    v_coord, v_label = read_vertices_txt(vert)

    coord, normal = load_mesh(stl)

    print("\n======", case_name, "======")

    scene_info(coord)

    # ------------------------------------------------
    # label assignment
    # ------------------------------------------------

    if len(coord) == len(v_coord):

        print("label mode: DIRECT COPY")

        segment = v_label

    else:

        print("label mode: NEAREST NEIGHBOR")

        segment = match_labels(v_coord, v_label, coord)


    sanity_check(segment)


    # ------------------------------------------------
    # save dataset
    # ------------------------------------------------

    for subdir, mode in zip(
        ["c5"],
        ["5"]
    ):

        out_dir = output_root / subdir / "Area_1" / case_name

        out_dir.mkdir(parents=True, exist_ok=True)

        if mode == "4":

            mask = segment > 0

            coord_out = coord[mask]
            normal_out = normal[mask]
            seg_out = remap(segment, mode)

        else:

            coord_out = coord
            normal_out = normal
            seg_out = remap(segment, mode)

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


# ------------------------------------------------
# main
# ------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--viz", action="store_true", help="保存每个 case 的三视图 png")
    parser.add_argument("--viz_max_points", type=int, default=300000, help="三视图渲染最多点数")
    parser.add_argument("--viz_point_size", type=float, default=0.2, help="三视图点大小")
    parser.add_argument(
        "--viz_boundary34",
        action="store_true",
        help="保存 class 3/4 交接处的局部三视图（triview_boundary34.png）",
    )
    parser.add_argument("--viz_boundary34_radius", type=float, default=0.005, help="class3/4 交接判定半径")
    parser.add_argument("--viz_boundary34_max_points_det", type=int, default=200000, help="boundary 检测阶段最多点数")
    parser.add_argument("--viz_boundary34_point_size", type=float, default=0.3, help="boundary 可视化点大小")

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    cases = [p for p in input_root.iterdir() if p.is_dir()]

    print("Total cases:", len(cases))

    for case in tqdm(cases):

        process_case(
            case,
            output_root,
            viz=args.viz,
            viz_max_points=args.viz_max_points,
            viz_point_size=args.viz_point_size,
            viz_boundary34=args.viz_boundary34,
            viz_boundary34_radius=args.viz_boundary34_radius,
            viz_boundary34_max_points_det=args.viz_boundary34_max_points_det,
            viz_boundary34_point_size=args.viz_boundary34_point_size,
        )


if __name__ == "__main__":

    main()