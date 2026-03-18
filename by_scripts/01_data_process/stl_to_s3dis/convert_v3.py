import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm


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
# process case
# ------------------------------------------------

def process_case(case_dir, output_root):

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
        ["c5", "c4", "c2"],
        ["5", "4", "2"]
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


# ------------------------------------------------
# main
# ------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    cases = [p for p in input_root.iterdir() if p.is_dir()]

    print("Total cases:", len(cases))

    for case in tqdm(cases):

        process_case(case, output_root)


if __name__ == "__main__":

    main()