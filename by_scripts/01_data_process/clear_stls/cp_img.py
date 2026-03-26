#!/usr/bin/env python3
"""
将 Area_1 下各子文件夹中的 PNG 复制到统一目录，并按子文件夹名重命名。

默认路径：
- src_root: D:/Projects/by_pointcept/data/03_train_data_aligned/version1/c5/Area_1
- dst_root: D:/Projects/by_pointcept/data/03_train_data_aligned/version1/c5/Area_1_view

规则：
1) 若子文件夹内存在 triview.png，优先复制它；
2) 否则若仅有一个 png，复制该 png；
3) 否则跳过并打印提示。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def pick_png(case_dir: Path) -> Path | None:
    triview = case_dir / "triview.png"
    if triview.exists():
        return triview

    pngs = sorted(case_dir.glob("*.png"))
    if len(pngs) == 1:
        return pngs[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_root",
        default="D:/Projects/by_pointcept/data/03_train_data_aligned/version4/c5/Area_1",
        help="源目录（包含多个病例子文件夹）",
    )
    ap.add_argument(
        "--dst_root",
        default="D:/Projects/by_pointcept/data/03_train_data_aligned/version4/c5/Area_1_view",
        help="目标目录（保存重命名后的 png）",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        raise SystemExit(f"src_root not found: {src_root}")

    total = 0
    copied = 0
    skipped = 0

    for case_dir in sorted(src_root.iterdir()):
        if not case_dir.is_dir():
            continue
        total += 1

        src_png = pick_png(case_dir)
        if src_png is None:
            print(f"[SKIP] {case_dir.name}: no unique png/triview.png")
            skipped += 1
            continue

        dst_png = dst_root / f"{case_dir.name}.png"
        shutil.copy2(src_png, dst_png)
        copied += 1

    print(f"[DONE] total_cases={total}, copied={copied}, skipped={skipped}, dst={dst_root}")


if __name__ == "__main__":
    main()

