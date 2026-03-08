#!/usr/bin/env python3
"""
从 00_raw_data 中挑选包含 vertices.txt 的子文件夹，拷贝到 01_raw_cls。
"""

import shutil
from pathlib import Path


def main():
    src_root = Path("/home/poker/workspace/11_third_projs/data/00_raw_data")
    dst_root = Path("/home/poker/workspace/11_third_projs/data/01_raw_cls")

    if not src_root.is_dir():
        raise SystemExit(f"源目录不存在: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    selected = []
    for item in src_root.iterdir():
        if not item.is_dir():
            continue
        vertices = item / "vertices.txt"
        if vertices.is_file():
            selected.append(item)

    print(f"找到 {len(selected)} 个包含 vertices.txt 的子文件夹")

    for folder in selected:
        dst = dst_root / folder.name
        if dst.exists():
            print(f"  跳过（已存在）: {folder.name}")
            continue
        try:
            shutil.copytree(folder, dst)
            print(f"  已拷贝: {folder.name}")
        except Exception as e:
            print(f"  失败 {folder.name}: {e}")

    print("完成。")


if __name__ == "__main__":
    main()
