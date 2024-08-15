import numpy as np
import matplotlib.pyplot as plt
from tools import db, eu_dis
import scipy
import os
import itertools

from multiprocessing.pool import ThreadPool
from tqdm import tqdm

slice_num = 1000


def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines


def Htmp_preprocess(Htmp, slice_samp_num, sc_num, ant_num, port_num):
    Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
    Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
    Htmp = np.transpose(Htmp, (0, 3, 2, 1))
    return Htmp


def dataprepare(cfg_path, inputdata_path, anchor_path):
    anchor_lines = read_slice_of_file(anchor_path, 0, slice_num)
    anchor_info = np.loadtxt(anchor_lines)
    slice_lines = read_slice_of_file(cfg_path, 1, 6)
    info = np.loadtxt(slice_lines)

    tot_samp_num = int(info[0])
    port_num = int(info[2])
    ant_num = int(info[3])
    sc_num = int(info[4])

    if not os.path.exists(
        r"/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos1/train_data_128_128_cnn/"
    ):
        os.makedirs(
            r"/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos1/train_data_128_128_cnn/",
            exist_ok=True,
        )
    slice_sam_num = 1
    load_data_slice = 0
    cnt = 0
    r_truncate = 256
    angle_resol_ele = 128
    angle_resol_azi = 128

    slice_all = read_slice_of_file(inputdata_path, 0, 20000)
    Htmp_all = np.loadtxt(slice_all)
    for slice_idx in anchor_info[:, 0]:
        slice_data_idx = int(slice_idx) - 1
        print("Data index:", slice_data_idx)
        print(f"--slice_idx:{cnt}---")
        Htmp = Htmp_preprocess(
            Htmp_all[slice_data_idx : slice_data_idx + 1, :],
            slice_sam_num,
            sc_num,
            ant_num,
            port_num,
        )
        # sub_idx = np.mod(slice_data_idx,slice_sam_num)z
        r_graph = np.fft.ifft(Htmp, axis=-1)
        r_graph = r_graph[:, :, :, :r_truncate]
        r_graph_antenna_array = (
            r_graph.transpose(0, 3, 1, 2)
            .reshape(1, r_truncate, 2, 2, 8, 4)
            .swapaxes(5, 4)
        )
        ra_graph = np.fft.fftshift(
            np.fft.fft2(r_graph_antenna_array, s=(angle_resol_ele, angle_resol_azi)),
            axes=(-2, -1),
        )
        ra_graph = ra_graph.transpose((0, 1, 4, 5, 2, 3)).reshape(
            (r_truncate, angle_resol_ele, angle_resol_azi, 4)
        )
        ra = db(ra_graph)
        ra = ra.astype(np.float32).squeeze()
        np.save(
            r"/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos1/train_data_128_128_cnn/"
            + str(cnt)
            + ".npy",
            ra,
        )
        cnt += 1


if __name__ == "__main__":
    cfg_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round2/Round2CfgData1.txt"
    inputdata_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round2/Round2InputData1.txt"
    anchor_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round2/Round2InputPos1.txt"
    dataprepare(cfg_path, inputdata_path, anchor_path)
