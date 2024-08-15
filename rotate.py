import numpy as np
import torch
# from typing import List
# from pathlib import Path
# import matplotlib.pyplot as plt
# import itertools


# def read_slice_of_file(file_path, start, end):
#     with open(file_path, "r") as file:
#         slice_lines = list(itertools.islice(file, start, end))
#     return slice_lines

from task_utils import easy_logger

logger = easy_logger()

def gen_rotated_matrix(angle: int):
    logger.warning('[red underline]::: NOTE: use torch instead of numpy, may cause some inconsistency [/red underline]' +
                   '[red underline]between numpy and torch, please check the results carefully. [/red underline]\n ' + 
                   '[red underline]注意：使用torch而不是numpy可能导致numpy和torch之间的不一致，决赛提交时使用的numpy进行旋转， [/red underline]' +
                   '[red underline]这里使用的是torch进行旋转，请仔细检查结果。经过测试，误差在[0.001, 0.1]之间。[/red underline] \n \n')
    
    angle = torch.tensor(angle)
    rad_ang = torch.deg2rad(angle)
    matrix =  torch.tensor(
                [
                    [torch.cos(rad_ang), torch.sin(rad_ang)],
                    [-torch.sin(rad_ang), torch.cos(rad_ang)],
                ],
                dtype=torch.float32
            )
    
    def _rotate(tensor: torch.Tensor):
        device = tensor.device
        assert tensor.ndim == 2 and tensor.size(-1) == 2, "Input tensor must be of shape (bs, 2)"
        
        return torch.einsum("ij,bj->bi", matrix.to(device), tensor)
    
    return _rotate


# if __name__ == f"__main__":
#     base_dirs = [
#         "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/results/raw3/Round3OutputPos2.txt"
#     ]
#     slice_num = 20000

#     ## -120 for pos1 and 120 for pos2
#     rotate_angles: List[float] = [120]
#     dst_data = []

#     for base_dir, angle in zip(base_dirs, rotate_angles):
#         truth_lines = read_slice_of_file(base_dir, 0, slice_num)
#         absolut_pos = np.loadtxt(truth_lines)[:, :].reshape(slice_num, 2)
#         rotate_mat = np.array(
#             [
#                 [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))],
#                 [-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
#             ],
#             dtype=float,
#         )

#         for i in range(absolut_pos.shape[0]):
#             dst_pt = rotate_mat @ (absolut_pos[i : i + 1, :].T)
#             dst_data.append(dst_pt)
#     with open("/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/results/rotated5/Round3OutputPos2.txt", "w",) as file:
#         for out in dst_data:
#             pos = f"%.4f %.4f\n" % (out[0][0], out[1][0])
#             file.writelines(pos)
