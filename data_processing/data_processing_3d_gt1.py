import numpy as np
import matplotlib.pyplot as plt
from tools import db, eu_dis
import scipy
import os
import itertools
import h5py
from tqdm import tqdm
import torch
from scipy.signal import wiener


def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines


def Htmp_preprocess(Htmp, slice_samp_num, sc_num, ant_num, port_num):
    Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
    Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
    Htmp = np.transpose(Htmp, (0, 3, 2, 1))
    return Htmp


def dataprepare(cfg_path, inputdata_path, save_path):
    slice_num = 20000
    slice_lines = read_slice_of_file(cfg_path, 1, 6)
    info = np.loadtxt(slice_lines)
    tot_samp_num = int(info[0])
    port_num = int(info[2])
    ant_num = int(info[3])
    sc_num = int(info[4])
    slice_sam_num = 1
    load_data_slice = 0
    cnt = 0
    r_truncate = 64
    angle_resol_ele = 64
    angle_resol_azi = 64

    h5 = h5py.File(save_path, "w")
    h5.create_dataset("data", (slice_num, 64, 64, 64, 4), dtype=np.float32, chunks=(1, 64, 64, 64, 4))
    slice_all = read_slice_of_file(inputdata_path, 0, 20000)
    Htmp_all = np.loadtxt(slice_all)
    for slice_idx in range(20000):
        slice_data_idx = int(slice_idx)
        # print('Data index:',slice_data_idx)
        print(f"--Data_idx:{slice_data_idx}---")
        print(f"--slice_idx:{cnt}--")
        Htmp = Htmp_preprocess(
            Htmp_all[slice_data_idx : slice_data_idx + 1, :],
            slice_sam_num,
            sc_num,
            ant_num,
            port_num,
        )
        Htmp = Htmp_preprocess(
            Htmp_all[slice_data_idx : slice_data_idx + 1, :],
            slice_sam_num,
            sc_num,
            ant_num,
            port_num,
        )
        for i in range(Htmp.shape[1]):
            for j in range(Htmp.shape[2]):
                Htmp[0, i, j, :] = wiener(np.real(Htmp[0, i, j, :])) + 1j * wiener(
                    np.imag(Htmp[0, i, j, :])
                )
        r_graph = np.fft.ifft(Htmp, n=64, axis=-1)
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
        ra_abs = db(ra_graph)
        data = (ra_abs).astype(np.float32).squeeze()
        th_data = torch.tensor(data).type(torch.float32).unsqueeze(0)
        th_data = th_data.permute(0, -1, 1, 2, 3)
        data = th_data[0].permute(1, 2, 3, 0).numpy()
        h5["data"][cnt] = data
        print("save [{}/{}] data".format(cnt, slice_num))
        cnt += 1
    h5.close()


if __name__ == "__main__":
    cfg_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round3/Round3CfgData1.txt"
    inputdata_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round3/Round3InputData1.txt"
    save_path = (
        r"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos1/test_gt_denoised_64.h5"
    )
    # anchor_path = r'/Data3/cao/ZiHanCao/datasets/huawei/Round3/Round3InputPos2.txt'
    # truth_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round0/Round0GroundTruth1.txt"
    dataprepare(cfg_path, inputdata_path, save_path)
