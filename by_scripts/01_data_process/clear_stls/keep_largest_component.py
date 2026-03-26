#!/usr/bin/env python3
"""
在 STL 网格中只保留“最大连通区域”（按三角面片连通性），其余区域只做删除。

为什么要这样做：
- 你的标注数据（如每个子目录下的 vertices.txt）依赖于“原始顶点信息/顺序”。
- 因此本脚本不会重排顶点，也不会对最大连通块做重建/重采样。
- 只会把不属于最大连通块的三角面片删除（默认不覆盖原始 stl，输出 *_largest.*）。

输出：
- 默认：<name>_largest.stl（便于继续沿用现有 pipeline）
- 可选：<name>_largest.ply（保留原 vertices 数组与索引，更利于与 vertices.txt 对齐验证）
- 可选：<name>_largest.kept_triangle_indices.txt / <name>_largest.kept_vertex_indices.txt（追溯用）
- 可选：vertices_largest.txt：默认按「到清理后网格表面的距离」筛选原始 vertices.txt 行（x y z label 原样保留），
  不再假设 STL 顶点索引与 vertices.txt 行号一一对应（STL triangle-soup 顶点数通常远大于标注点数）。

依赖：
- open3d（你们现有脚本已在用）

用法示例：
  python by_scripts/01_data_process/clear_stls/keep_largest_component.py ^
    --root D:\\Projects\\by_pointcept\\data\\01_raw_cls --write_ply 1

  # 只统计不写文件
  python by_scripts/01_data_process/clear_stls/keep_largest_component.py ^
    --root D:\\Projects\\by_pointcept\\data\\01_raw_cls --dry_run 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import open3d as o3d


def _iter_stl_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.stl")
    else:
        yield from root.glob("*.stl")


def _choose_largest_cluster(
    triangles: np.ndarray,
    triangle_clusters: np.ndarray,
    cluster_n_triangles: np.ndarray,
    cluster_area: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    返回：
    - best_cluster_id
    - kept_triangle_indices (1D, int)
    - kept_vertex_indices (1D, int, unique/sorted)
    """
    if triangles.size == 0:
        raise ValueError("empty triangles")

    # 以“面积最大”为主，三角面数作为次级 tie-breaker
    best_cluster_id = -1
    best_area = -1.0
    best_tri_count = -1
    best_tri_idx = None
    best_vert_idx = None

    # cluster id 通常是 [0..K-1]
    cluster_ids = np.arange(int(triangle_clusters.max()) + 1, dtype=np.int32)
    for cid in cluster_ids:
        tri_idx = np.flatnonzero(triangle_clusters == cid).astype(np.int64)
        if tri_idx.size == 0:
            continue
        tcnt = int(cluster_n_triangles[cid]) if cid < len(cluster_n_triangles) else int(tri_idx.size)
        area = float(cluster_area[cid]) if cid < len(cluster_area) else float("-inf")

        if (area > best_area) or (area == best_area and tcnt > best_tri_count):
            best_cluster_id = int(cid)
            best_area = area
            best_tri_count = tcnt
            best_tri_idx = tri_idx
            best_vert_idx = np.unique(triangles[tri_idx].reshape(-1)).astype(np.int64)

    if best_cluster_id < 0 or best_tri_idx is None or best_vert_idx is None:
        raise RuntimeError("failed to pick largest connected component")

    return best_cluster_id, best_tri_idx, best_vert_idx


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        # path compress
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return int(p)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = self.rank[ra] + 1


def _triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    # area = 0.5 * ||(v1-v0) x (v2-v0)||
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _approx_merge_vertices(vertices: np.ndarray, eps: float) -> np.ndarray:
    """
    用量化哈希做“近似焊接”：
    返回 merged_id: shape (N,), 每个原顶点映射到一个合并后的 id（0..M-1）
    """
    if eps <= 0:
        return np.arange(vertices.shape[0], dtype=np.int64)

    # 量化到 eps 网格
    q = np.round(vertices / eps).astype(np.int64)
    merged_id = np.empty(vertices.shape[0], dtype=np.int64)

    table: Dict[Tuple[int, int, int], int] = {}
    next_id = 0
    for i in range(q.shape[0]):
        key = (int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        mid = table.get(key)
        if mid is None:
            mid = next_id
            table[key] = mid
            next_id += 1
        merged_id[i] = mid

    return merged_id


def keep_largest_connected_component_mesh(
    mesh: o3d.geometry.TriangleMesh,
    weld_eps: float = 0.02,
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    """
    仅删除小连通块三角面片，不重排顶点。

    Returns:
      out_mesh: 顶点数组保持不变，三角面片为最大连通块的子集
      kept_triangle_indices: 保留下来的原三角面索引
      kept_vertex_indices: 最大连通块涉及到的原顶点索引（unique/sorted）
    """
    if mesh is None:
        raise ValueError("mesh is None")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError("mesh has no vertices/triangles")

    # 关键点：
    # STL 常见是 triangle soup（顶点几乎不共享），且坐标可能有微小浮点差。
    # 所以这里用“近似焊接（容差 weld_eps）”在索引层面建立连通关系：
    # - 不改动原顶点数组/顺序
    # - 只用于判断哪些三角面属于同一连通区域
    merged_id = _approx_merge_vertices(vertices.astype(np.float64, copy=False), eps=float(weld_eps))
    tri_m = merged_id[triangles]  # (T,3) 合并后的顶点 id

    # 建三角面邻接：共享一条边（无向边）
    # 用边到“第一个出现的三角面”索引的映射来 union
    dsu = _DSU(triangles.shape[0])
    edge_owner: Dict[Tuple[int, int], int] = {}
    for t in range(tri_m.shape[0]):
        a, b, c = int(tri_m[t, 0]), int(tri_m[t, 1]), int(tri_m[t, 2])
        edges = ((a, b), (b, c), (c, a))
        for u, v in edges:
            if u == v:
                continue
            if u > v:
                u, v = v, u
            key = (u, v)
            prev = edge_owner.get(key)
            if prev is None:
                edge_owner[key] = t
            else:
                dsu.union(prev, t)

    # 统计每个 component 的面积和三角数
    areas = _triangle_areas(vertices.astype(np.float64, copy=False), triangles.astype(np.int64, copy=False))
    root = np.array([dsu.find(i) for i in range(triangles.shape[0])], dtype=np.int64)

    comp_area: Dict[int, float] = {}
    comp_count: Dict[int, int] = {}
    for i in range(root.shape[0]):
        r = int(root[i])
        comp_area[r] = comp_area.get(r, 0.0) + float(areas[i])
        comp_count[r] = comp_count.get(r, 0) + 1

    # 选面积最大的 component（tie-breaker: 三角数）
    best_r = None
    best_area = -1.0
    best_cnt = -1
    for r, a in comp_area.items():
        c = comp_count[r]
        if (a > best_area) or (a == best_area and c > best_cnt):
            best_r = r
            best_area = a
            best_cnt = c
    if best_r is None:
        raise RuntimeError("failed to compute connected components")

    kept_tri_idx = np.flatnonzero(root == int(best_r)).astype(np.int64)
    kept_vert_idx = np.unique(triangles[kept_tri_idx].reshape(-1)).astype(np.int64)

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(vertices)  # 不动顶点
    out_mesh.triangles = o3d.utility.Vector3iVector(triangles[kept_tri_idx])  # 只删面
    out_mesh.compute_vertex_normals()

    return out_mesh, kept_tri_idx, kept_vert_idx


def _write_indices_txt(path: Path, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in indices.tolist():
            f.write(f"{int(i)}\n")


def _export_vertices_txt_subset(
    src_vertices_txt: Path,
    dst_vertices_txt: Path,
    kept_vertex_indices: np.ndarray,
) -> None:
    """
    从原始 vertices.txt 中按“顶点索引”抽取子集，保持每行的 x y z label 完全不变。

    约定：
    - 第 1 行是注释/表头
    - 第 2 行是数量（整数）
    - 从第 3 行开始每一行对应一个顶点（行号-3 即顶点索引）

    注意：仅当 STL 顶点顺序与 vertices.txt 行一一对应时才正确；STL triangle-soup 时请勿使用。
    """
    if not src_vertices_txt.exists():
        return

    lines = src_vertices_txt.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=False)
    if len(lines) < 3:
        return

    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    kept = kept[(kept >= 0) & (kept < (len(lines) - 2))]
    kept = np.unique(kept)

    out_lines: List[str] = []
    out_lines.append(lines[0])
    out_lines.append(str(int(kept.size)))
    for vid in kept.tolist():
        out_lines.append(lines[2 + int(vid)])

    dst_vertices_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_vertices_txt.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _distances_to_mesh_surface(mesh_legacy: o3d.geometry.TriangleMesh, points_xyz: np.ndarray) -> np.ndarray:
    """点到三角网格表面的距离，shape (N,)。"""
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    t = o3d.core.Tensor(points_xyz.astype(np.float32))
    d = scene.compute_distance(t)
    return np.asarray(d.numpy()).reshape(-1).astype(np.float64)


def _export_vertices_txt_by_surface_distance(
    src_vertices_txt: Path,
    dst_vertices_txt: Path,
    mesh_largest: o3d.geometry.TriangleMesh,
    dist_thresh: float,
    batch: int = 50000,
) -> None:
    """
    用「标注点到清理后网格表面的距离」筛选 vertices.txt 行；每行字符串原样保留（含 x y z label）。
    """
    if not src_vertices_txt.exists():
        return

    lines = src_vertices_txt.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=False)
    if len(lines) < 3:
        return

    data_lines = lines[2:]
    n = len(data_lines)
    coords = np.zeros((n, 3), dtype=np.float64)
    valid_row = np.zeros(n, dtype=bool)
    for i, ln in enumerate(data_lines):
        s = ln.strip().split()
        if len(s) < 4:
            continue
        try:
            coords[i, 0] = float(s[0])
            coords[i, 1] = float(s[1])
            coords[i, 2] = float(s[2])
            valid_row[i] = True
        except Exception:
            continue

    if not np.any(valid_row):
        return

    dists = np.full(n, np.inf, dtype=np.float64)
    idx_valid = np.flatnonzero(valid_row)
    for s in range(0, idx_valid.size, batch):
        chunk_idx = idx_valid[s : s + batch]
        dists[chunk_idx] = _distances_to_mesh_surface(mesh_largest, coords[chunk_idx])

    keep_mask = valid_row & (dists <= float(dist_thresh))
    kept_lines = [data_lines[i] for i in np.flatnonzero(keep_mask)]

    out_lines: List[str] = []
    out_lines.append(lines[0])
    out_lines.append(str(len(kept_lines)))
    out_lines.extend(kept_lines)

    dst_vertices_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_vertices_txt.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(
        f"[VERTICES] surface distance: kept {len(kept_lines)}/{n} (thresh={dist_thresh}) -> {dst_vertices_txt}"
    )


def process_one_stl(
    stl_path: Path,
    root: Path,
    out_root: Path,
    suffix: str,
    write_ply: bool,
    write_index_txt: bool,
    export_vertices_txt: bool,
    vertices_match_mode: str,
    vertices_dist_thresh: float,
    weld_eps: float,
    dry_run: bool,
) -> None:
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print(f"[SKIP] empty mesh: {stl_path}")
        return

    out_mesh, kept_tri_idx, kept_vert_idx = keep_largest_connected_component_mesh(mesh, weld_eps=weld_eps)

    total_tri = len(mesh.triangles)
    kept_tri = int(kept_tri_idx.size)
    print(f"[OK] {stl_path} | triangles kept {kept_tri}/{total_tri} | vertices-in-component {int(kept_vert_idx.size)}")

    if dry_run:
        return

    rel = stl_path.relative_to(root)
    out_dir = out_root / rel.parent
    out_stl = out_dir / (stl_path.stem + suffix + stl_path.suffix)

    out_stl.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(out_stl), out_mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"failed to write: {out_stl}")

    if write_ply:
        out_ply = out_stl.with_suffix(".ply")
        ok = o3d.io.write_triangle_mesh(str(out_ply), out_mesh, write_ascii=True)
        if not ok:
            raise RuntimeError(f"failed to write: {out_ply}")

    if write_index_txt:
        base = out_stl.with_suffix("")
        _write_indices_txt(base.with_suffix(".kept_triangle_indices.txt"), kept_tri_idx)
        _write_indices_txt(base.with_suffix(".kept_vertex_indices.txt"), kept_vert_idx)

    if export_vertices_txt:
        src_vtxt = stl_path.with_name("vertices.txt")
        dst_vtxt = out_dir / "vertices_largest.txt"
        if vertices_match_mode == "index":
            _export_vertices_txt_subset(src_vtxt, dst_vtxt, kept_vert_idx)
        else:
            _export_vertices_txt_by_surface_distance(
                src_vtxt, dst_vtxt, out_mesh, dist_thresh=vertices_dist_thresh
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=str(Path("data/01_raw_cls")),
        help="数据根目录（其下各子文件夹包含 *.stl 和 vertices.txt）",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="",
        help="统一输出目录（默认：<root>_largest，且保留原子目录结构）",
    )
    ap.add_argument("--recursive", type=int, default=1, help="是否递归扫描子目录（1/0）")
    ap.add_argument("--only_name", type=str, default="femur.stl", help="只处理该文件名（空字符串表示不过滤）")
    ap.add_argument("--suffix", type=str, default="_largest", help="输出文件后缀（inplace=0 时生效）")
    ap.add_argument("--write_ply", type=int, default=1, help="是否同时写出 PLY（1/0）")
    ap.add_argument("--write_index_txt", type=int, default=1, help="是否写出保留的三角/顶点索引列表（1/0）")
    ap.add_argument(
        "--export_vertices_txt",
        type=int,
        default=1,
        help="是否在输出目录写出 vertices_largest.txt（x y z label 行内容不改）（1/0）",
    )
    ap.add_argument(
        "--vertices_match_mode",
        type=str,
        default="surface",
        choices=["surface", "index"],
        help=(
            "vertices_largest.txt 如何筛选："
            "surface=默认，到清理后 femur_largest 网格表面距离<=阈值；"
            "index=用 STL 顶点索引当 vertices.txt 行号（仅当两者索引空间一致时可用）"
        ),
    )
    ap.add_argument(
        "--vertices_dist_thresh",
        type=float,
        default=1.0,
        help="vertices_match_mode=surface 时保留点的距离阈值（单位与坐标一致，如 mm）",
    )
    ap.add_argument("--weld_eps", type=float, default=0.02, help="近似焊接容差（单位同 STL 坐标）")
    ap.add_argument("--dry_run", type=int, default=0, help="只统计不写文件（1/0）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    out_root = Path(args.out_root) if str(args.out_root).strip() else Path(str(root) + "_largest")
    if not bool(args.dry_run):
        out_root.mkdir(parents=True, exist_ok=True)

    stls = list(_iter_stl_files(root, recursive=bool(args.recursive)))
    if str(args.only_name).strip():
        only = str(args.only_name).strip().lower()
        stls = [p for p in stls if p.name.lower() == only]
    if not stls:
        print(f"[WARN] no stl found under: {root}")
        return

    for p in stls:
        try:
            process_one_stl(
                stl_path=p,
                root=root,
                out_root=out_root,
                suffix=args.suffix,
                write_ply=bool(args.write_ply),
                write_index_txt=bool(args.write_index_txt),
                export_vertices_txt=bool(args.export_vertices_txt),
                vertices_match_mode=str(args.vertices_match_mode),
                vertices_dist_thresh=float(args.vertices_dist_thresh),
                weld_eps=float(args.weld_eps),
                dry_run=bool(args.dry_run),
            )
        except Exception as e:
            print(f"[FAIL] {p} | {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

