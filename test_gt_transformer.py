from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import threading
from collections import OrderedDict
import queue
import argparse
from rich.progress import track
from tqdm import tqdm
from enum import Enum

import Transformer3D
from code3.zihan.rotate_fields import gen_rotated_matrix
from task_utils import easy_logger, getMemInfo
from tools import read_slice_of_file

logger = easy_logger()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--pos_n', type=int, default=1, help='posN')
    parser.add_argument('--loading_strategy', type=str, default='ALL', help='loading strategy', choices=['ALL', 'PART', 'ONFLY'])
    
    args = parser.parse_args()
    
    return args
    
args = get_args()
    
## configures
device = args.device
torch.cuda.set_device(device)
posN = args.pos_n
acIndices = {
    1: [0, 500],
    2: [500, 1000],
    3: [1000, 1500],
}
angle = {1: -120, 2: 120, 3: 0}[posN]
rotateFn = gen_rotated_matrix(angle) if angle != 0 else lambda x: x

class LoadingStrategy(Enum):
    ALL = 0
    PART = 1
    ONFLY = 2
    
loadingStrategy = {"ALL": LoadingStrategy.ALL, 
                   "PART": LoadingStrategy.PART, 
                   "ONFLY": LoadingStrategy.ONFLY}[args.loading_strategy]

# weightPath = {
#     1: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformerP1/ep_1_iter199.pth",
#     2: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformerP2/ep_1_iter99.pth",
#     3: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformer3dP3/ep_1_iter199.pth",
# }[posN]
weightPath = {
    1: "ckpts/R3P1.pth",
    2: "ckpts/R3P2.pth",
    3: "ckpts/R3P3.pth",
}[posN]
acFile = f"h5file/R3Anchors/Round3PosAll.txt"
acFile2 = f"h5file/R3Anchors/Round3InputPos{posN}.txt"
infFile = f"results/Round3OutputPos{posN}.txt"
h5FilePath = f"h5file/R3P{posN}"
logger.info(f"use {posN=}\n\n")
logger.info(f'inference file will be saved at {infFile}')

factor = 100
SNum = 20000
sNumR1 = 0
SNumTest = 0
AcLen = 500


class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        # test_dir = f"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos{pos_n}/test_gt_64.h5"
        # testPath = f"/Data4/exps/dataset/Round3Pos{posN}_test_gt_64.h5"
        testPath = f"/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/h5file/Round3Pos{posN}_test_gt_64_normed.h5"
        testFile = h5py.File(testPath, "r")
        logger.warning('ready to load all data into RAM, may cause OOM.')
        logger.warning('and this may consume 80G RAM or more.')
        self.testData = testFile["data"][:]
        logger.info(f"chunks: {self.testData.chunks}")
        # global_idx_to_local = {}
        
        # for i in range(total_len):
        #     global_idx_to_local[i] = (0, i)

        # self.global_idx_to_local = global_idx_to_local
        # data_files = [test_file]
        # self.data_files = data_files

        # self.datalist = []

    def __len__(self) -> int:
        return self.testData.shape[0]

    def __getitem__(self, index):
        # global_idx = index
        # dataset_idx, idx = self.global_idx_to_local[global_idx]

        # rand_data = self.data_files[dataset_idx][idx]

        # return rand_data
        
        return torch.from_numpy(self.testData[index])#.cuda(device=device, non_blocking=True)


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


class TestDataset2(Dataset):
    def __init__(self, cache_size=1000):
        super().__init__()
        test_dir = f"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos{posN}/test_gt_64.h5"
        # test_dir = f"/Data4/exps/dataset/Round3Pos{pos_n}_test_gt_64.h5"
        self.test_file = h5py.File(test_dir, "r", swmr=True)["data"]
        print(f'chunks: {self.test_file.chunks}')
        self.total_len = self.test_file.shape[0]
        
        self.cache = LRUCache(cache_size)
        self.load_queue = queue.Queue()
        
        # Start loader threads
        self.stop_event = threading.Event()
        self.loader_threads = [
            threading.Thread(target=self._loader_worker)
            for _ in range(8)  # You can adjust the number of threads
        ]
        for t in self.loader_threads:
            t.start()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        data = self.cache.get(index)
        if data is not None:
            return data
        
        self.load_queue.put(index)
        while True:
            data = self.cache.get(index)
            if data is not None:
                return data
            # Small sleep to reduce CPU usage while waiting
            threading.Event().wait(0.001)

    def _loader_worker(self):
        while not self.stop_event.is_set():
            try:
                index = self.load_queue.get(timeout=1)
                data = self.test_file[index]
                self.cache.put(index, data)
            except queue.Empty:
                continue

    def __del__(self):
        self.stop_event.set()
        for t in self.loader_threads:
            t.join()
        self.test_file.close()


class TestSplitedDataset(Dataset):
    """
    BaiduCloudDisk only support <= 20G file, so we need to split the file.
    To ensure the test, we load the all of the list of splited datasets in the directory.
    """
    def __init__(self, h5_dir: str):
        super().__init__()
        self.h5_dir = h5_dir
        assert Path(h5_dir).exists(), 'h5_dir not exists.'

        self.h5_files = sorted(list(Path(h5_dir).glob('*.h5')), key=lambda x: int(x.stem[:2]))
        self.h5_files = [h5py.File(f, 'r') for f in self.h5_files]
        # 2000 per h5 files, total number of test file is 20000
        assert len(self.h5_files) == 10, 'the number of h5 files must be 10.'
        
        self.total_len = 20000
        assert self.total_len == 20000, 'the total number of samples must be 20000.'
        
        if loadingStrategy in (LoadingStrategy.PART, LoadingStrategy.ALL):
            logger.warning('ready to load all data into RAM, may cause OOM.')
            logger.warning('and this may consume [red]80G[/red] RAM or more.')
        else:
            logger.warning('ready to load data on fly, this strategy is really slow, use it when the RAM is limited.')
        
        if loadingStrategy == LoadingStrategy.PART:
            self.data = None
            self.sub_h5_idx = -1
        elif loadingStrategy == LoadingStrategy.ALL:
            self.data = self._all_load_h5_file()
        elif loadingStrategy == LoadingStrategy.ONFLY:
            self.data = self.h5_files[0]['data']
            self.sub_h5_idx = 0
        else:
            raise ValueError('invalid loading strategy.')
        
    def _part_load_h5_file(self, idx):
        # h5 file index 0 -> 0 ~ 2000
        # h5 file index 1 -> 2000 ~ 4000
        sub_h5_max_idx = (self.sub_h5_idx + 1) * 2000  # max value of the data range
        if idx < sub_h5_max_idx and idx != 0:
            in_h5_idx = idx - self.sub_h5_idx * 2000
            return self.data[in_h5_idx]
        else:
            self.sub_h5_idx += 1
            f = self.h5_files[self.sub_h5_idx]
            assert f["data"].shape[0] >= 2000, 'the number of samples in each h5 file must be 2000.'
            logger.info(f'========================[{self.sub_h5_idx}]/[{len(self.h5_files)}]===========================')
            logger.info(f'loading 2000 samples from [{self.sub_h5_idx}] h5 file...')
            # self.data = f['data'][:2000]  # 2000 is the number of samples in each h5 file
            self.data = self._per_sample_fast_loading(f['data'])
            f.close()
            getMemInfo(logger)
            
            in_h5_idx = idx - self.sub_h5_idx * 2000
            return self.data[in_h5_idx]
        
    def _per_sample_fast_loading(self, h5_data_file):
        n_samples = 2000
        data_lst = []
        for i in tqdm(range(n_samples), total=n_samples,
                      desc='loading sub-h5 file per sample ...', leave=False):
            data_lst.append(h5_data_file[i])
            
        return np.stack(data_lst, axis=0)
        
    def _all_load_h5_file(self):
        data_lst = []
        for f in tqdm(self.h5_files, total=len(self.h5_files), desc='Loading h5 files...'):
            data_lst.append(f['data'][:])
            getMemInfo(logger)
            logger.info(f'loading data from [{f.filename}] h5 file...')
        data = np.concatenate(data_lst, axis=0)
        
        return data
    
    def _on_fly_load_h5_file(self, idx):
        sub_h5_max_idx = (self.sub_h5_idx + 1) * 2000  # max value of the data range
        if idx < sub_h5_max_idx:
            in_h5_idx = idx - self.sub_h5_idx * 2000
            return self.data[in_h5_idx]
        else:
            logger.info(f'next h5 file: idx {self.sub_h5_idx + 1}')
            self.h5_files[self.sub_h5_idx].close()
            self.sub_h5_idx += 1
        
        f = self.h5_files[self.sub_h5_idx]['data']
        assert f.shape[0] >= 2000, 'the number of samples in each h5 file must be 2000.'
        self.data = f
        
        in_h5_idx = idx - self.sub_h5_idx * 2000
        return self.data[in_h5_idx]
        
    def __len__(self):
        ## hard coded length
        return self.total_len
    
    def __getitem__(self, index) -> torch.Tensor:
        if loadingStrategy == LoadingStrategy.PART:
            data = self._part_load_h5_file(index)
            data = torch.from_numpy(data)
        elif loadingStrategy == LoadingStrategy.ALL:
            data = torch.from_numpy(self.data[index])
        else:
            data = self._on_fly_load_h5_file(index)
            data = torch.from_numpy(data)
        
        return data
    
    

class Runner:
    @classmethod
    def inference(cls) -> None:
        anchor = read_slice_of_file(acFile, *acIndices[posN])
        acAll = np.loadtxt(anchor).reshape(AcLen, 2)
        acPos = torch.from_numpy(acAll)
        ac2 = read_slice_of_file(acFile2, 0, AcLen)
        ac2All = np.loadtxt(ac2).reshape(AcLen, 3)
        index = ac2All[:, 0]
        index = list(np.int32(index))

        # state_dict = torch.load(weight_path)
        # model = generate_model_3d(10)
        # model.load_state_dict(state_dict)

        # for n, m in model.named_modules():
        #     if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
        #         m.track_running_stats = False
        #         logger.info(f"{n} is set to track_running_stats=False")

        model = Transformer3D.generate_model(101)
        model.load_state_dict(torch.load(weightPath))
        # model = torch.load(weight_path)
        logger.info("model loaded.")
        model.eval()

        logger.info("loading dataset...")
        # test_dataset = TestDataset()
        test_dataset = TestSplitedDataset(h5FilePath)
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=32,
            num_workers=0,
            # prefetch_factor=200,
            # pin_memory=True,
            # pin_memory_device=device,
        )

        model = model.cuda(device=device)

        output = []
        logger.info("model predict...")
        file = open(infFile, "w")
        
        k = 0
        bs = test_loader.batch_size
        with torch.no_grad():
            error = 0
            for i, xx in track(enumerate(test_loader), total=len(test_loader),
                               description='Inference ...'):
                xx = xx.cuda(device=device, non_blocking=True)

                xy = model(xx)
                xy = rotateFn(xy)
                for ii in range(xx.shape[0]):
                    gi = bs * i + ii

                    # if in the AC list
                    if gi + 1 in index:
                        logger.info(f"{k}: {acPos[k,:]}")
                        pos = acPos[k, :]
                        pos = rotateFn(pos[None].to(xy)).detach().cpu().numpy()[0]
                        pos = "%.4f %.4f\n" % (pos[0], pos[1])
                        file.write(pos)
                        k += 1
                    else:
                        logger.info(f"model pred: {xy[ii]}")
                        pos = "%.4f %.4f\n" % (xy[ii][0], xy[ii][1])
                        file.write(pos)
                        
                        
                        # output.append(xy[ii].detach().cpu().numpy())

                # if i + 1 in index[:]:
                #     logger.info(f"{k}: {anchor_pos[k,:]}")
                #     output.append(anchor_pos[k : k + 1, :])
                #     k += 1
                # else:
                #     xy = model(data)
                #     logger.info(f"model pred: {xy}")
                #     output.append(xy.detach().cpu().numpy())
        # return output
        file.close()


if __name__ == "__main__":
    output = Runner.inference()
    
    # ds = TestSplitedDataset("/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/h5file/R3P1")
    # dl = DataLoader(ds, batch_size=32, num_workers=0, pin_memory=True)
    # for i, data in track(enumerate(dl), total=len(dl), description='Inference ...'):
    #     pass