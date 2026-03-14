import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# -----------------------------------------
# compute boundary ratio
# -----------------------------------------

def compute_boundary_ratio(coord, segment, k=16):

    nn = NearestNeighbors(n_neighbors=k).fit(coord)

    _, idx = nn.kneighbors(coord)

    neighbor_label = segment[idx]

    boundary = (neighbor_label != segment[:, None]).any(axis=1)

    return boundary.mean()


# -----------------------------------------
# curvature estimation
# -----------------------------------------

def compute_curvature(coord, k=16):

    nn = NearestNeighbors(n_neighbors=k).fit(coord)

    _, idx = nn.kneighbors(coord)

    curvatures = []

    for i in range(len(coord)):

        pts = coord[idx[i]]

        cov = np.cov(pts.T)

        eigvals = np.linalg.eigvalsh(cov)

        curvature = eigvals[0] / eigvals.sum()

        curvatures.append(curvature)

    return np.mean(curvatures)


# -----------------------------------------
# process single scene
# -----------------------------------------

def process_scene(scene_path):

    coord = np.load(scene_path / "coord.npy")
    segment = np.load(scene_path / "segment.npy")

    boundary = compute_boundary_ratio(coord, segment)

    curvature = compute_curvature(coord)

    labels, counts = np.unique(segment, return_counts=True)

    return {
        "points": len(coord),
        "boundary": boundary,
        "curvature": curvature,
        "labels": dict(zip(labels.tolist(), counts.tolist()))
    }


# -----------------------------------------
# worker wrapper
# -----------------------------------------

def worker(scene_path):

    try:
        return process_scene(scene_path)

    except Exception as e:

        print("error:", scene_path, e)

        return None


# -----------------------------------------
# main
# -----------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", required=True)

    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count()
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    scenes = list(dataset_root.glob("Area_*/*"))

    print("scenes:", len(scenes))
    print("workers:", args.workers)

    results = []

    with Pool(args.workers) as pool:

        for r in tqdm(
            pool.imap(worker, scenes),
            total=len(scenes)
        ):
            if r:
                results.append(r)

    # -----------------------------------------
    # aggregate results
    # -----------------------------------------

    total_points = sum(r["points"] for r in results)

    boundary_mean = np.mean([r["boundary"] for r in results])

    curvature_mean = np.mean([r["curvature"] for r in results])

    label_counts = {}

    for r in results:

        for k, v in r["labels"].items():

            label_counts[k] = label_counts.get(k, 0) + v

    # -----------------------------------------
    # print summary
    # -----------------------------------------

    print("\n===== DATASET SUMMARY =====")

    print("Total points:", total_points)

    print("\nLabel distribution:")

    for k in sorted(label_counts):

        print(f"class {k}: {label_counts[k]}")

    print("\nLabel ratio:")

    for k in sorted(label_counts):

        print(f"class {k}: {label_counts[k] / total_points:.4f}")

    print("\nBoundary ratio (mean):", boundary_mean)

    print("Curvature mean:", curvature_mean)

    print("\n===== DATASET HEALTH CHECK =====")

    if boundary_mean < 0.01:
        print("WARNING: boundary ratio very low")

    if curvature_mean < 0.001:
        print("WARNING: curvature very low")


if __name__ == "__main__":
    main()


"""
python dataset_quality_check_parallel.py \
--dataset_root dataset_s3dis/c5 \
--workers 16
"""