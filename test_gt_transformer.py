import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
from rich.progress import track

import Transformer3D
from code3.zihan.data_processing.data_processing_3d_gt import read_slice_of_file
from rotate import gen_rotated_matrix
from task_utils import easy_logger

logger = easy_logger()

## configures
device = "cuda:0"
torch.cuda.set_device(device)
posN = 1
acIndices = {
    1: [0, 500],
    2: [500, 1000],
    3: [1000, 1500],
}
angle = {1: -120, 2: 120, 3: 0}[posN]
rotateFn = gen_rotated_matrix(angle) if angle != 0 else lambda x: x
# weight_path = {1: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformerP1/ep_3_iter299.pth",
#     # 1: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformerP1/ep_2_iter99(used).pth",
#                2: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformer3dP2/ep_1_iter299(used).pth",
#                3: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformer3dP3/ep_4_iter99(used).pth"}[pos_n]
# weight_path = {
#     1: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet10/P1_state_dict.pth",
#     2: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet10/P2_state_dict.pth",
#     3: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet10/P3_state_dict.pth"
# }[pos_n]
weightPath = {
    1: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformerP1/ep_1_iter199.pth",
    2: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformerP2/ep_1_iter99.pth",
    3: "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/finetuneTransformer3dP3/ep_1_iter199.pth",
}[posN]
acFile = f"/Data3/cao/ZiHanCao/datasets/huawei/round3Pos123P3_all.txt"
acFile2 = f"/Data3/cao/ZiHanCao/datasets/huawei/Round3/Round3InputPos{posN}.txt"
infFile = f"/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/results/Round3OutputPos{posN}.txt"
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
        self.testData = testFile["data"]#[:]
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


import threading
from collections import OrderedDict
import queue

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
        test_dataset = TestDataset()
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
                # time.sleep(0.05)
                
                # data = (data - MEAN.to(data)) / STD.to(data)
                # data = data.permute(0, 4, 1, 2, 3)
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
