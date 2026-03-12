#!/usr/bin/env python3
"""
S3DIS数据集可视化脚本
用于可视化指定场景在不同区域（Area_1, Area_2等）的分割结果
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

# 为了在 WSL/headless 环境下也能保存图片，这里使用 matplotlib 进行离线渲染
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# S3DIS数据集的13个类别及其颜色映射
S3DIS_CLASSES = [
    'ceiling',      # 0
    'floor',        # 1
    'wall',         # 2
    'beam',         # 3
    'column',       # 4
    'window',       # 5
    'door',         # 6
    'table',        # 7
    'chair',        # 8
    'sofa',         # 9
    'bookcase',     # 10
    'board',        # 11
    'clutter'       # 12
]

# 为每个类别定义直观的颜色（RGB，范围0-1）
def generate_color_palette(num_classes=13):
    """生成颜色调色板，使用更直观的颜色映射"""
    import colorsys
    
    # 为S3DIS的13个类别定义更直观的颜色
    class_colors = [
        (0.8, 0.8, 0.8),  # ceiling - 浅灰色
        (0.4, 0.2, 0.1),  # floor - 棕色
        (0.9, 0.9, 0.7),  # wall - 米黄色
        (0.5, 0.5, 0.5),  # beam - 灰色
        (0.3, 0.3, 0.3),  # column - 深灰色
        (0.2, 0.6, 0.9),  # window - 蓝色
        (0.6, 0.3, 0.1),  # door - 深棕色
        (0.8, 0.4, 0.2),  # table - 橙色
        (0.2, 0.8, 0.4),  # chair - 绿色
        (0.8, 0.2, 0.6),  # sofa - 粉色
        (0.4, 0.2, 0.6),  # bookcase - 紫色
        (0.9, 0.7, 0.2),  # board - 黄色
        (0.9, 0.2, 0.2),  # clutter - 红色
    ]
    
    colors = np.zeros((num_classes, 3))
    for i in range(min(num_classes, len(class_colors))):
        colors[i] = np.array(class_colors[i])
    
    # 如果类别数超过预定义颜色，使用HSV生成
    if num_classes > len(class_colors):
        for i in range(len(class_colors), num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[i] = np.array(rgb)
    
    return colors


def load_scene_data(data_root, area_name, scene_name):
    """
    加载指定区域和场景的数据
    
    Args:
        data_root: 数据集根目录
        area_name: 区域名称，如 'Area_1'
        scene_name: 场景名称，如 'conferenceRoom_1'
    
    Returns:
        dict: 包含coord, color, segment等数据的字典，如果不存在则返回None
    """
    scene_path = os.path.join(data_root, area_name, scene_name)
    
    if not os.path.exists(scene_path):
        return None
    
    data = {}
    
    # 加载坐标
    coord_path = os.path.join(scene_path, 'coord.npy')
    if os.path.exists(coord_path):
        data['coord'] = np.load(coord_path)
    else:
        return None
    
    # 加载分割标签
    segment_path = os.path.join(scene_path, 'segment.npy')
    if os.path.exists(segment_path):
        data['segment'] = np.load(segment_path)
        # 如果segment是(N, 1)形状，转换为(N,)
        if len(data['segment'].shape) > 1 and data['segment'].shape[1] == 1:
            data['segment'] = data['segment'].flatten()
    else:
        return None
    
    # 加载颜色（可选）
    color_path = os.path.join(scene_path, 'color.npy')
    if os.path.exists(color_path):
        data['color'] = np.load(color_path)
        # 如果color是uint8格式，转换为0-1范围
        if data['color'].dtype == np.uint8:
            data['color'] = data['color'].astype(np.float32) / 255.0
    
    return data


def visualize_segmentation(
    coord,
    segment,
    color_palette,
    title="Point Cloud Segmentation",
    save_path=None,
    show_window=False,
):
    """
    可视化分割结果
    
    Args:
        coord: 点云坐标 (N, 3)
        segment: 分割标签 (N,)
        color_palette: 颜色调色板 (num_classes, 3)
        title: 窗口标题
        save_path: 保存图片的路径（可选）
    """
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    
    # 根据分割标签分配颜色
    num_points = len(coord)
    colors = np.zeros((num_points, 3))
    
    # 获取唯一的标签
    unique_labels = np.unique(segment)
    
    for label in unique_labels:
        if label >= 0 and label < len(color_palette):
            mask = segment == label
            colors[mask] = color_palette[int(label)]
        else:
            # 对于超出范围的标签，使用灰色
            mask = segment == label
            colors[mask] = [0.5, 0.5, 0.5]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化
    print(f"\n可视化: {title}")
    print(f"点云数量: {num_points}")
    print(f"分割类别: {unique_labels}")
    print(f"类别名称: {[S3DIS_CLASSES[int(l)] if int(l) < len(S3DIS_CLASSES) else f'unknown_{l}' for l in unique_labels]}")
    
    if save_path:
        # 使用 matplotlib 进行离线渲染，兼容 WSL/headless 环境
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 采样一部分点以避免极大点云导致绘制过慢（例如超过 300k 点时随机采样）
        max_points = 300000
        num_points = coord.shape[0]
        if num_points > max_points:
            idx = np.random.choice(num_points, max_points, replace=False)
            coord_plot = coord[idx]
            colors_plot = colors[idx]
        else:
            coord_plot = coord
            colors_plot = colors

        ax.scatter(
            coord_plot[:, 0],
            coord_plot[:, 1],
            coord_plot[:, 2],
            c=colors_plot,
            s=0.2,
            linewidth=0,
        )
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"已保存图片到: {save_path}")

        # 如需在有图形环境时额外弹出交互窗口
        if show_window:
            try:
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name=title,
                    width=1920,
                    height=1080,
                )
            except Exception as e:
                print(f"尝试打开 Open3D 窗口失败（可能是 WSL 无图形环境）: {e}")
    else:
        # 仍然保留交互式 Open3D 可视化，用于有图形界面的环境
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=1920,
            height=1080
        )


def visualize_with_original_color(coord, color, segment, color_palette, title="Point Cloud with Segmentation"):
    """
    同时显示原始颜色和分割结果（可选）
    """
    # 创建两个点云：一个显示原始颜色，一个显示分割结果
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(coord)
    if color is not None:
        pcd_original.colors = o3d.utility.Vector3dVector(color)
    
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(coord)
    
    num_points = len(coord)
    colors = np.zeros((num_points, 3))
    unique_labels = np.unique(segment)
    
    for label in unique_labels:
        if label >= 0 and label < len(color_palette):
            mask = segment == label
            colors[mask] = color_palette[int(label)]
        else:
            # 对于超出范围的标签，使用灰色
            mask = segment == label
            colors[mask] = [0.5, 0.5, 0.5]
    
    pcd_seg.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化（可以切换查看）
    print(f"\n可视化: {title}")
    print(f"点云数量: {num_points}")
    print(f"分割类别: {unique_labels}")
    
    # 先显示分割结果
    o3d.visualization.draw_geometries(
        [pcd_seg],
        window_name=f"{title} - Segmentation",
        width=1920,
        height=1080
    )


def find_scene_in_areas(data_root, scene_name):
    """
    在所有区域中查找指定场景
    
    Args:
        data_root: 数据集根目录
        scene_name: 场景名称
    
    Returns:
        list: 找到的场景路径列表，格式为 [(area_name, scene_path), ...]
    """
    found_scenes = []
    
    # 遍历所有Area文件夹
    for area_dir in sorted(os.listdir(data_root)):
        area_path = os.path.join(data_root, area_dir)
        if not os.path.isdir(area_path) or not area_dir.startswith('Area_'):
            continue
        
        scene_path = os.path.join(area_path, scene_name)
        if os.path.exists(scene_path):
            found_scenes.append((area_dir, scene_path))
    
    return found_scenes


def main():
    parser = argparse.ArgumentParser(description='可视化S3DIS数据集的分割结果')
    parser.add_argument('--scene', type=str, required=True,
                        help='场景名称，例如: conferenceRoom_1')
    parser.add_argument('--data_root', type=str, 
                        default='/root/autodl-fs/s3dis',
                        help='数据集根目录路径')
    parser.add_argument('--area', type=str, default='Area_1',
                        help='指定区域（如Area_1），如果不指定则显示所有找到的区域')
    parser.add_argument('--show_original', action='store_true',
                        help='同时显示原始颜色（如果可用）')
    parser.add_argument('--save_image', type=str, default=None,
                        help='保存可视化结果为图片（指定保存路径，如: output.png）')
    parser.add_argument(
        '--show_window',
        action='store_true',
        help='在保存图片的同时尝试弹出 Open3D 交互式窗口（需要宿主有图形环境）',
    )
    
    args = parser.parse_args()
    
    # 生成颜色调色板
    color_palette = generate_color_palette(len(S3DIS_CLASSES))
    
    # 查找场景
    if args.area:
        # 只查找指定区域
        areas_to_check = [args.area]
    else:
        # 查找所有区域
        found_scenes = find_scene_in_areas(args.data_root, args.scene)
        if not found_scenes:
            print(f"错误: 未找到场景 '{args.scene}' 在任何区域中")
            return
        
        areas_to_check = [area for area, _ in found_scenes]
        print(f"找到场景 '{args.scene}' 在以下区域: {', '.join(areas_to_check)}")
    
    # 可视化每个区域
    for area_name in areas_to_check:
        print(f"\n{'='*60}")
        print(f"处理区域: {area_name}")
        print(f"{'='*60}")
        
        data = load_scene_data(args.data_root, area_name, args.scene)
        
        if data is None:
            print(f"警告: 在 {area_name} 中未找到场景 {args.scene} 或数据不完整")
            continue
        
        coord = data['coord']
        segment = data['segment']
        color = data.get('color', None)
        
        # 检查数据维度
        if len(coord.shape) != 2 or coord.shape[1] != 3:
            print(f"错误: 坐标数据格式不正确，期望形状为 (N, 3)，实际为 {coord.shape}")
            continue
        
        if len(segment.shape) != 1:
            print(f"错误: 分割标签格式不正确，期望形状为 (N,)，实际为 {segment.shape}")
            continue
        
        if len(coord) != len(segment):
            print(f"错误: 坐标和分割标签数量不匹配: {len(coord)} vs {len(segment)}")
            continue
        
        # 可视化
        title = f"{args.scene} - {area_name}"
        
        # 确定保存路径
        save_path = None
        if args.save_image:
            # 如果指定了保存路径，为每个区域生成不同的文件名
            base_name = os.path.splitext(args.save_image)[0]
            ext = os.path.splitext(args.save_image)[1] or '.png'
            save_path = f"{base_name}_{area_name}{ext}"
        
        if args.show_original and color is not None:
            visualize_with_original_color(coord, color, segment, color_palette, title)
        else:
            visualize_segmentation(
                coord,
                segment,
                color_palette,
                title,
                save_path=save_path,
                show_window=args.show_window,
            )
        
        print(f"完成 {area_name} 的可视化")


if __name__ == '__main__':
    main()

