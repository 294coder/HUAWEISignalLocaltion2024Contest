import numpy as np
import torch

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
