import numpy as np
import torch

from task_utils import easy_logger

logger = easy_logger()


def gen_rotated_matrix(angle: int):
    """
    This method generates a rotation matrix based on the given angle and returns a function that can be used to rotate a tensor.

    Parameters:
    - angle (int): The angle of rotation in degrees.

    Returns:
    - _rotate (function): A function that takes a tensor of shape (bs, 2) and rotates it using the generated rotation matrix.

    Note:
    - This method uses torch instead of numpy for rotation, which may cause some inconsistency between numpy and torch. Please check the results carefully.
    - The rotation is performed using torch.cos and torch.sin functions, and the resulting matrix is of dtype torch.float32.
    - The input tensor must be of shape (bs, 2), where bs is the batch size.
    - The rotation is performed using torch.einsum function.

    Example usage:
    rotate_fn = gen_rotated_matrix(45)
    rotated_tensor = rotate_fn(input_tensor)
    """
    
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
