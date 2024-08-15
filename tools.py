import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def db(data):
    return 20 * np.log10(np.abs(data))


def eu_dis(ue1_pos, ue2_pos):
    return np.sqrt(
        (ue1_pos[:, 0] - ue2_pos[:, 0]) ** 2
        + (ue1_pos[:, 1] - ue2_pos[:, 1]) ** 2
        + (ue1_pos[:, 2] - ue2_pos[:, 2] ** 2)
    )
