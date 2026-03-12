#!/usr/bin/env python3
"""
将 femur_stl 数据转换为 S3DIS 格式

输入：
- STL 网格文件（femur.stl, tibia_fibula.stl）
- 顶点分割标注文件（vertices.txt）

输出：
- S3DIS 格式数据（coord.npy, segment.npy, color.npy, normal.npy）
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 少数类设置：默认将 2、3 视为少数类
MINORITY_CLASSES = (2, 3)
MINORITY_OVERSAMPLE_FACTOR = 0  # >0 时，对少数类做点级 oversample
MINORITY_JITTER_STD = 0.0       # oversample 时的坐标扰动标准差


def read_vertices_txt(vertices_path):
    """
    读取 vertices.txt 文件
    
    Args:
        vertices_path: vertices.txt 文件路径
        
    Returns:
        coords: (N, 3) 顶点坐标
        labels: (N,) 分割标签（如果 Constant 字段是标签）
    """
    with open(vertices_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过注释行和数量行
    data_lines = []
    for line in lines[2:]:  # 跳过 "//X Y Z Constant" 和数量行
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 4:
                data_lines.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    
    data = np.array(data_lines)
    coords = data[:, :3]  # X, Y, Z
    labels = data[:, 3].astype(np.int32)  # Constant 字段作为标签
    
    return coords, labels


def load_stl_mesh(stl_path):
    """
    加载 STL 网格文件
    
    Args:
        stl_path: STL 文件路径
        
    Returns:
        mesh: Open3D TriangleMesh 对象
    """
    if not os.path.exists(stl_path):
        return None
    
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0:
        return None
    
    return mesh


def mesh_to_point_cloud(mesh, num_points=None, use_vertices=True):
    """
    将网格转换为点云
    
    Args:
        mesh: Open3D TriangleMesh 对象
        num_points: 目标点云数量（如果为 None，使用所有顶点）
        use_vertices: 如果为 True，使用顶点；如果为 False，采样面片
        
    Returns:
        coords: (N, 3) 点云坐标
        normals: (N, 3) 法向量
    """
    if use_vertices:
        coords = np.array(mesh.vertices).astype(np.float32)
        # 计算顶点法向量
        mesh.compute_vertex_normals()
        normals = np.array(mesh.vertex_normals).astype(np.float32)
    else:
        # 采样点云
        if num_points is None:
            num_points = len(mesh.vertices)
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        coords = np.array(pcd.points).astype(np.float32)
        normals = np.array(pcd.normals).astype(np.float32)
    
    return coords, normals


def match_labels_to_points(vertices_coords, vertices_labels, point_coords, distance_threshold=0.01):
    """
    将 vertices.txt 中的标签匹配到点云
    
    Args:
        vertices_coords: (M, 3) vertices.txt 中的坐标
        vertices_labels: (M,) vertices.txt 中的标签
        point_coords: (N, 3) 点云坐标
        distance_threshold: 匹配距离阈值
        
    Returns:
        labels: (N,) 点云标签
    """
    # 使用 Open3D 的 KDTree 进行最近邻搜索
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(vertices_coords)
    kdtree = o3d.geometry.KDTreeFlann(pcd_ref)
    
    # 初始化标签为 -1（未匹配）
    labels = np.full(len(point_coords), -1, dtype=np.int32)
    
    # 对每个点进行半径搜索 + 多票表决，提升标签稳定性
    radius = float(distance_threshold)
    for i, point in enumerate(point_coords):
        [k, idx, dist] = kdtree.search_radius_vector_3d(point, radius)
        if k > 0:
            neighbor_labels = vertices_labels[idx]
            # 去掉无效标签（<0），做多数表决
            valid = neighbor_labels[neighbor_labels >= 0]
            if len(valid) > 0:
                labels[i] = np.bincount(valid).argmax()
    
    return labels


def process_patient_folder(patient_folder, output_root, area_name, distance_threshold=0.01, include_tibia=False):
    """
    处理单个患者文件夹
    
    Args:
        patient_folder: 患者文件夹路径
        output_root: 输出根目录
        area_name: 区域名称（如 Area_1）
        distance_threshold: 标签匹配距离阈值
        include_tibia: 是否包含 tibia_fibula.stl（默认: False，只处理 femur.stl）
        
    Returns:
        bool: 是否成功处理
    """
    patient_folder = Path(patient_folder)
    patient_name = patient_folder.name
    
    # 查找 STL 文件
    femur_stl = patient_folder / "femur.stl"
    tibia_stl = patient_folder / "tibia_fibula.stl"
    vertices_txt = patient_folder / "vertices.txt"
    
    if not vertices_txt.exists():
        # 在多线程环境下，错误信息会在主线程统一显示
        return False
    
    # 读取顶点标注
    try:
        vertices_coords, vertices_labels = read_vertices_txt(vertices_txt)
    except Exception as e:
        # 在多线程环境下，错误信息会在主线程统一显示
        return False
    
    # 收集所有 STL 文件的点云
    all_coords = []
    all_normals = []
    
    stl_files = []
    if femur_stl.exists():
        stl_files.append(("femur", femur_stl))
    if include_tibia and tibia_stl.exists():
        stl_files.append(("tibia_fibula", tibia_stl))
    
    if not stl_files:
        # 在多线程环境下，错误信息会在主线程统一显示
        return False
    
    # 先收集所有点云
    for part_name, stl_path in stl_files:
        # 加载网格
        mesh = load_stl_mesh(stl_path)
        if mesh is None:
            # 在多线程环境下，错误信息会在主线程统一显示
            continue
        
        # 转换为点云
        coords, normals = mesh_to_point_cloud(mesh, use_vertices=True)
        
        if len(coords) == 0:
            continue
        
        all_coords.append(coords)
        all_normals.append(normals)
    
    if len(all_coords) == 0:
        # 在多线程环境下，错误信息会在主线程统一显示
        return False
    
    # 合并所有点云
    coord = np.concatenate(all_coords, axis=0)
    normal = np.concatenate(all_normals, axis=0)
    
    # 匹配标签到点云
    # 如果数量相同，先尝试快速匹配（检查是否顺序一致）
    if len(coord) == len(vertices_coords):
        # 检查坐标是否按顺序匹配
        coord_diff = np.abs(coord - vertices_coords)
        max_diff = np.max(coord_diff)
        
        if max_diff < distance_threshold:
            # 坐标按顺序匹配，直接使用标签
            segment = vertices_labels.copy()
        else:
            # 坐标不完全匹配，使用最近邻匹配
            segment = match_labels_to_points(
                vertices_coords, vertices_labels, coord, distance_threshold
            )
    else:
        # 数量不同，使用最近邻匹配
        segment = match_labels_to_points(
            vertices_coords, vertices_labels, coord, distance_threshold
        )

    # 针对少数类（如 2、3），对未标注点做一次“膨胀”匹配，提升少数类覆盖率
    try:
        minority_mask_vertices = np.isin(vertices_labels, MINORITY_CLASSES)
        if minority_mask_vertices.any():
            pcd_min = o3d.geometry.PointCloud()
            pcd_min.points = o3d.utility.Vector3dVector(
                vertices_coords[minority_mask_vertices]
            )
            kdt_min = o3d.geometry.KDTreeFlann(pcd_min)
            # 半径略放大，只用于少数类扩张
            radius_min = float(distance_threshold) * 2.0
            unknown_idx = np.where(segment < 0)[0]
            for i in unknown_idx:
                [k, idx, dist] = kdt_min.search_radius_vector_3d(coord[i], radius_min)
                if k > 0:
                    # 邻域中若存在少数类顶点，则将该点吸附为对应标签
                    min_labels = vertices_labels[minority_mask_vertices][idx]
                    valid = min_labels[min_labels >= 0]
                    if len(valid) > 0:
                        segment[i] = np.bincount(valid).argmax()
    except Exception:
        # 若少数类扩张过程出错，不影响整体流程
        pass
    
    # 如果所有标签都是0或-1，根据部位分配默认标签
    if np.all(segment <= 0):
        # 默认所有点都是 femur (0)
        segment[:] = 0
    
    # 对少数类做点级 oversample，提高模型训练中对少数类的采样频率
    if MINORITY_OVERSAMPLE_FACTOR > 0:
        minority_mask = np.isin(segment, MINORITY_CLASSES)
        num_minority = int(minority_mask.sum())
        if num_minority > 0:
            base_coords = coord[minority_mask]
            base_normals = normal[minority_mask]
            base_labels = segment[minority_mask]

            repeat = int(max(1, MINORITY_OVERSAMPLE_FACTOR))
            aug_coords = np.repeat(base_coords, repeat, axis=0)
            aug_normals = np.repeat(base_normals, repeat, axis=0)
            aug_labels = np.repeat(base_labels, repeat, axis=0)

            if MINORITY_JITTER_STD > 0.0:
                noise = np.random.normal(
                    scale=float(MINORITY_JITTER_STD),
                    size=aug_coords.shape,
                ).astype(np.float32)
                aug_coords = aug_coords + noise

            coord = np.concatenate([coord, aug_coords], axis=0)
            normal = np.concatenate([normal, aug_normals], axis=0)
            segment = np.concatenate([segment, aug_labels], axis=0)
    
    # 生成颜色（基于标签）
    color = np.ones((len(coord), 3), dtype=np.uint8) * 128  # 默认灰色
    
    # 创建输出目录
    scene_name = patient_name.replace(" ", "_").replace(".", "_")
    output_dir = Path(output_root) / area_name / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    try:
        np.save(output_dir / "coord.npy", coord.astype(np.float32))
        np.save(output_dir / "segment.npy", segment.astype(np.int16))
        np.save(output_dir / "normal.npy", normal.astype(np.float32))
        np.save(output_dir / "color.npy", color.astype(np.uint8))
        
        return True
    except Exception as e:
        # 在多线程环境下，错误信息会在主线程统一显示
        return False


def main():
    parser = argparse.ArgumentParser(description='将 femur_stl 数据转换为 S3DIS 格式')
    parser.add_argument('--input_root', type=str,
                        default=r'C:\win_projs\third\beiyi3\data\beiyi\raw\femur_stl',
                        help='输入数据根目录')
    parser.add_argument('--output_root', type=str,
                        default=r'C:\win_projs\third\beiyi3\data\beiyi\beiyi_s3dis',
                        help='输出数据根目录')
    parser.add_argument('--area_name', type=str, default='Area_1',
                        help='区域名称（如 Area_1）。对于医学数据，建议使用单个 Area，如 Area_1')
    parser.add_argument('--split_areas', action='store_true',
                        help='是否将数据划分到多个 Area（Area_1, Area_2, ...）。默认 False，所有数据放在一个 Area')
    parser.add_argument('--num_areas', type=int, default=1,
                        help='如果 split_areas=True，划分到多少个 Area（默认: 1，即所有数据在一个 Area）')
    parser.add_argument('--train_test_split', action='store_true',
                        help='按 10:1 比例自动划分训练集和测试集（训练集: Area_1, 测试集: Area_5）')
    parser.add_argument('--train_ratio', type=float, default=10.0/11.0,
                        help='训练集比例（默认: 10/11，即 10:1 划分）')
    parser.add_argument('--distance_threshold', type=float, default=0.01,
                        help='标签匹配距离阈值')
    parser.add_argument('--include_tibia', action='store_true',
                        help='是否包含 tibia_fibula.stl（默认: False，只处理 femur.stl）')
    parser.add_argument('--patient_id', type=str, default=None,
                        help='只处理指定的患者ID（文件夹名）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子，用于可重复的数据划分（默认: 42）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='并行处理的线程数（默认: 4，设置为 1 则单线程处理）')
    parser.add_argument(
        '--minority_classes',
        type=str,
        default='2,3',
        help='少数类标签列表，例如 "2,3"（默认: 2,3）',
    )
    parser.add_argument(
        '--minority_oversample_factor',
        type=int,
        default=2,
        help='对少数类点进行 oversample 的重复倍数（默认: 2，设置为 0 关闭）',
    )
    parser.add_argument(
        '--minority_jitter_std',
        type=float,
        default=0.3,
        help='少数类 oversample 时坐标扰动的标准差（同点云单位，默认: 0.3）',
    )
    
    args = parser.parse_args()

    # 配置全局少数类设置
    global MINORITY_CLASSES, MINORITY_OVERSAMPLE_FACTOR, MINORITY_JITTER_STD
    try:
        MINORITY_CLASSES = tuple(
            int(x.strip()) for x in args.minority_classes.split(",") if x.strip() != ""
        )
    except Exception:
        MINORITY_CLASSES = (2, 3)
    MINORITY_OVERSAMPLE_FACTOR = max(0, int(args.minority_oversample_factor))
    MINORITY_JITTER_STD = max(0.0, float(args.minority_jitter_std))
    
    input_root = Path(args.input_root)
    if not input_root.exists():
        print(f"错误: 输入目录不存在: {input_root}")
        return
    
    # 获取所有患者文件夹
    patient_folders = [f for f in input_root.iterdir() if f.is_dir()]
    
    if args.patient_id:
        patient_folders = [f for f in patient_folders if f.name == args.patient_id]
    
    if len(patient_folders) == 0:
        print("未找到患者文件夹")
        return
    
    # 确定 Area 划分策略
    if args.train_test_split:
        # 按 10:1 比例划分训练集和测试集
        import random
        random.seed(args.random_seed)
        random.shuffle(patient_folders)
        
        num_train = int(len(patient_folders) * args.train_ratio)
        train_folders = patient_folders[:num_train]
        test_folders = patient_folders[num_train:]
        
        area_assignments = {}
        for pf in train_folders:
            area_assignments[pf] = 'Area_1'  # 训练集
        for pf in test_folders:
            area_assignments[pf] = 'Area_5'  # 测试集（S3DIS 通常用 Area_5 作为测试集）
        
        print(f"按 10:1 比例划分训练集和测试集:")
        print(f"  训练集 (Area_1): {len(train_folders)} 个患者")
        print(f"  测试集 (Area_5): {len(test_folders)} 个患者")
        print(f"  随机种子: {args.random_seed}")
    elif args.split_areas and args.num_areas > 1:
        # 将患者分配到不同的 Area
        area_assignments = {}
        for i, patient_folder in enumerate(patient_folders):
            area_idx = (i % args.num_areas) + 1
            area_name = f"Area_{area_idx}"
            area_assignments[patient_folder] = area_name
        print(f"将 {len(patient_folders)} 个患者划分到 {args.num_areas} 个 Area")
    else:
        # 所有患者放在同一个 Area
        area_assignments = {pf: args.area_name for pf in patient_folders}
        print(f"所有 {len(patient_folders)} 个患者将放在 {args.area_name}")
    
    print(f"输出目录: {args.output_root}")
    print(f"并行线程数: {args.num_workers}")
    
    # 处理每个患者（多线程）
    success_count = 0
    failed_patients = []
    
    if args.num_workers <= 1:
        # 单线程处理（用于调试）
        for patient_folder in tqdm(patient_folders, desc="处理患者"):
            area_name = area_assignments[patient_folder]
            if process_patient_folder(patient_folder, args.output_root, area_name, 
                                     args.distance_threshold, args.include_tibia):
                success_count += 1
            else:
                failed_patients.append(patient_folder.name)
    else:
        # 多线程处理
        lock = threading.Lock()
        
        def process_single_patient(patient_folder):
            """处理单个患者的包装函数"""
            area_name = area_assignments[patient_folder]
            try:
                result = process_patient_folder(
                    patient_folder, args.output_root, area_name, 
                    args.distance_threshold, args.include_tibia
                )
                return result, patient_folder.name, None
            except Exception as e:
                # 捕获异常，返回错误信息
                return False, patient_folder.name, str(e)
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_patient = {
                executor.submit(process_single_patient, pf): pf 
                for pf in patient_folders
            }
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(patient_folders), desc="处理患者") as pbar:
                for future in as_completed(future_to_patient):
                    result, patient_name, error_msg = future.result()
                    if result:
                        success_count += 1
                    else:
                        failed_patients.append((patient_name, error_msg))
                    pbar.update(1)
    
    print(f"\n处理完成: {success_count}/{len(patient_folders)} 个患者成功处理")
    if failed_patients:
        print(f"\n失败的患者 ({len(failed_patients)} 个):")
        for item in failed_patients[:10]:  # 只显示前10个
            if isinstance(item, tuple):
                name, error = item
                if error:
                    print(f"  - {name}: {error}")
                else:
                    print(f"  - {name}")
            else:
                print(f"  - {item}")
        if len(failed_patients) > 10:
            print(f"  ... 还有 {len(failed_patients) - 10} 个失败的患者")
    
    # 打印 Area 分布统计
    if args.train_test_split or (args.split_areas and args.num_areas > 1):
        area_counts = Counter(area_assignments.values())
        print("\nArea 分布:")
        for area_name in sorted(area_counts.keys()):
            print(f"  {area_name}: {area_counts[area_name]} 个患者")


if __name__ == '__main__':
    main()

