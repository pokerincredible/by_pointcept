"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from .defaults import DefaultDataset
from .builder import DATASETS

# 合并格式 data.npy: (N, 7) float32 -> [x,y,z, nx,ny,nz, segment]
DATA_NPY_COORD_SLICE = slice(0, 3)
DATA_NPY_NORMAL_SLICE = slice(3, 6)
DATA_NPY_SEGMENT_COL = 6


@DATASETS.register_module()
class S3DISDataset(DefaultDataset):
    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data_npy_path = os.path.join(data_path, "data.npy")

        if os.path.isfile(data_npy_path):
            return self._load_merged_data_npy(data_path, data_npy_path, idx)
        return super().get_data(idx)

    def _load_merged_data_npy(self, data_path, data_npy_path, idx):
        """从合并的 data.npy 加载：格式 (N,7) = coord(3) + normal(3) + segment(1)。"""
        raw = np.load(data_npy_path)
        if raw.ndim != 2 or raw.shape[1] != 7:
            raise ValueError(
                f"data.npy expected shape (N, 7), got {getattr(raw, 'shape', 'unknown')}"
            )

        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        data_dict = {
            "coord": raw[:, DATA_NPY_COORD_SLICE].astype(np.float32),
            "normal": raw[:, DATA_NPY_NORMAL_SLICE].astype(np.float32),
            "segment": raw[:, DATA_NPY_SEGMENT_COL].reshape(-1).astype(np.int32),
            "instance": np.ones(raw.shape[0], dtype=np.int32) * -1,
            "name": name,
            "split": split,
        }
        return data_dict
