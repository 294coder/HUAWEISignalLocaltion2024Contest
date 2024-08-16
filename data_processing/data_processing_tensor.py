import numpy as np
import itertools
import h5py
import torch
from tqdm import tqdm, trange
from scipy.signal import wiener

def db(data):
    return 20 * np.log10(np.abs(data))

def eu_dis(ue1_pos, ue2_pos):
    return np.sqrt(
        (ue1_pos[:, 0] - ue2_pos[:, 0]) ** 2
        + (ue1_pos[:, 1] - ue2_pos[:, 1]) ** 2
        + (ue1_pos[:, 2] - ue2_pos[:, 2] ** 2)
    )

def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines


def Htmp_preprocess(Htmp, slice_samp_num, sc_num, ant_num, port_num):
    Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
    Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
    Htmp = np.transpose(Htmp, (0, 3, 2, 1))
    return Htmp


def data_transforming_train(cfg_path, slice_num, anchor_path, inputdata_path, save_path):
    MEAN = np.array([-5.091097, -5.1093035, -5.0822234, -5.0828133], dtype=torch.float32)[None, None, None]
    STD = np.array()([5.861487, 5.879711, 5.8663826, 5.875907], dtype=torch.float32)[None, None, None]
    
    anchor_lines = read_slice_of_file(anchor_path, 0, slice_num)
    anchor_info = np.loadtxt(anchor_lines)
    slice_lines = read_slice_of_file(cfg_path, 1, 6)
    info = np.loadtxt(slice_lines)
    tot_samp_num = int(info[0])
    port_num = int(info[2])
    ant_num = int(info[3])
    sc_num = int(info[4])
    slice_sam_num = 1
    load_data_slice = 0
    cnt = 0
    r_truncate = 128
    angle_resol_ele = 64
    angle_resol_azi = 64

    h5 = h5py.File(save_path, "w")
    h5.create_dataset(
        "data", (slice_num, 64, angle_resol_ele, angle_resol_azi, 4), dtype=np.float32
    )
    slice_all = read_slice_of_file(inputdata_path, 0, 4000)
    Htmp_all = np.loadtxt(slice_all)
    for slice_idx in tqdm(anchor_info[:, 0], total=len(anchor_info[:, 0])):
        slice_data_idx = int(slice_idx) - 1
        print(f"index : {slice_data_idx}")
        print(f"mini idx: {cnt}")
        Htmp = Htmp_preprocess(
            Htmp_all[slice_data_idx : slice_data_idx + 1, :],
            slice_sam_num,
            sc_num,
            ant_num,
            port_num,
        )
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
        ra_abs = db(ra_graph)
        data = (ra_abs).astype(np.float32).squeeze()
        th_data = torch.tensor(data).type(torch.float32).unsqueeze(0)
        th_data = th_data.permute(0, -1, 1, 2, 3)
        th_data = torch.nn.functional.interpolate(
            th_data, size=[64, 64, 64], mode="trilinear", align_corners=False
        )
        data = th_data[0].permute(1, 2, 3, 0).numpy()  # [64, 64, 64, 4]
        ## normalize
        data = (data - MEAN) / STD
        
        h5["data"][cnt] = data
        print("save [{}/{}] data".format(cnt, slice_num))
        cnt += 1
    h5.close()
    
def data_transforming_test(cfg_path, inputdata_path, save_path) -> None:
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
    for slice_idx in trange(20000):
        slice_data_idx = int(slice_idx)
        print(f"index : {slice_data_idx}")
        print(f"mini idx: {cnt}")
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
    cfg_path = r"/Data4/exps/dataset/Round0CfgData3.txt"
    inputdata_path = r"/Data4/exps/dataset/Round0InputData3.txt"
    save_path = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round0Pos3/all_old64.h5"
    anchor_path = r"/Data4/exps/dataset/Round0InputPos3.txt"
    # truth_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round0/Round0GroundTruth1.txt"
    data_transforming_train(cfg_path, anchor_path, inputdata_path, save_path)
