from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import h5py
import argparse
from tqdm import tqdm
from rich.progress import track
# import torch_ema
from ema_pytorch import EMA

# from time_resnet3d import generate_model as generate_resnet3d
# from models.transformer3d import segformer3d_base
# from models.uniformer import uniformer_base, uniformer_small
from models.convnext3d import generate_model as generate_resnet3d
from task_utils import EasyProgress, easy_logger, catch_any_error, get_virtual_memory
from data_processing import read_slice_of_file

logger = easy_logger()

finetune = True
pretrained_weight = "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet101/P2_ep_1_iter199.pth"
if finetune:
    assert pretrained_weight != "", "pretrained_weight should not be empty"
prefix_weight_name = "TransformerP2"
if finetune:
    prefix_weight_name = "finetune" + prefix_weight_name

# array_count = 4
# angle_dim = 128
# range_dim = 150
factor = 200 if not finetune else 1000
slice_num = 450
slice_num_r1 = 0
slice_num_r2 = 2760 if not finetune else 0
slice_num_test = 50
# Inputpos.txt中样本个数 即真值

#MEAN = np.array([-13.381914 , -13.39852  , -13.3906765, -13.38957], dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
#STD = np.array([4.7102904, 4.687373 , 4.6926985, 4.7124653], dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, np.newaxis]

MEAN = np.array([-5.091097 , -5.1093035, -5.0822234, -5.0828133], dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
STD = np.array([5.861487 , 5.879711 , 5.8663826, 5.875907 ], dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
# to tensor
MEAN = torch.tensor(MEAN)
STD = torch.tensor(STD)

logger.info("loading select set ...")
select_set = np.zeros(
    (int((slice_num + slice_num_r1 + slice_num_r2) * factor), 1), dtype=np.int32
)
cnt = 0
for i in track(
    range(int((slice_num + slice_num_r1 + slice_num_r2) * factor)),
    total=int((slice_num + slice_num_r1 + slice_num_r2) * factor),
):
    select_set[cnt, 0] = np.random.randint(0, slice_num + slice_num_r1 + slice_num_r2)
    cnt += 1
print("ok")
# select_set = np.zeros((int((slice_num+slice_num_r1) * (slice_num+slice_num_r1 - 1) / 2), 1), dtype=np.int32)
# cnt = 0
# for i in track(range(slice_num_r1+slice_num), total=slice_num+slice_num_r1):
#     for j in range(i + 1, slice_num+slice_num_r1):
#         select_set[cnt, 0] = i
#         cnt += 1
# print('ok')


def set_ema_model_params_with_keys(ema_model_params: "dict[str, list[torch.Tensor] | int | float]", 
                                   keys: "list[str]",
                                   keys_set: list[str]=['shadow_params']):
    """set ema model parameters with keys

    Args:
        ema_model_params (dict[str, list[torch.Tensor] | int | float]): ema model parameters
        keys (list[str]): keys

    Returns:
        dict: ema model parameters with keys
    """
    logger = easy_logger()
    
    if not isinstance(keys, list):
        keys = list(keys)
    
    ema_model_params_with_keys = OrderedDict()
    for k in ema_model_params.keys():
        if k in keys_set and k in ema_model_params:
            logger.info(f'set ema_model {k} params with keys')
            params = ema_model_params[k]
            assert params is not None
            assert len(params) == len(keys), "ema_model_params and keys should have the same length"
            
            _params = OrderedDict()
            for mk, p in zip(keys, params):
                _params[mk] = p
                
            ema_model_params_with_keys[k] = _params
        elif k not in keys_set and k in ema_model_params:
            ema_model_params_with_keys[k] = ema_model_params[k]
            
    return ema_model_params_with_keys

class CosineAnnealingWarmRestartsReduce(CosineAnnealingWarmRestarts):
    def __init__(
        self, opt: "optim.Optimizer", T_0, T_mult=1, lr_mult=1, eta_min=0, last_epoch=-1
    ):
        self.opt = opt
        self.lr_mult = lr_mult
        super().__init__(opt, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        super().step(epoch)

        if self.T_cur == self.T_i - 1 and self.last_epoch != 0:
            # reduce the base lr
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] *= self.lr_mult
                self.base_lrs[i] = max(self.base_lrs[i], self.eta_min)


class TrainDataset(Dataset):
    def __init__(self):
        logger.warning(
            '[red] load all files into RAM, be careful with the memory usage [/red]'
        )
        
        data_dir1 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos2/train_new64.h5"
        file1 = h5py.File(data_dir1, "r")['data'][:]
        get_virtual_memory(logger)
        
        if not finetune:
            # data_dir2 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round1Pos2/all_new64.h5"
            # file2 = h5py.File(data_dir2, "r")['data'][:]
            # get_virtual_memory(logger)
            
            # data_dir3 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round1Pos3/all_new64.h5"
            # file3 = h5py.File(data_dir3, "r")['data'][:]
            # get_virtual_memory(logger)
            
            data_dir4 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos1/train.h5"
            file4 = h5py.File(data_dir4, "r")['data'][:]
            get_virtual_memory(logger)
            
            data_dir5 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos2/train.h5"
            file5 = h5py.File(data_dir5, "r")['data'][:]
            get_virtual_memory(logger)
            
            data_dir6 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos3/train.h5"
            file6 = h5py.File(data_dir6, "r")['data'][:]
            get_virtual_memory(logger)
        
            anchor_path = r"/Data3/cao/ZiHanCao/datasets/huawei/round3Pos123P3.txt"
            anchor_path2 = r'/Data3/cao/ZiHanCao/datasets/huawei/round2Pos123P3.txt'
            
            # cat file1 and file2 together
            # global idx -> (dataset idx, idx in dataset)
            global_idx_to_local = {}
            total_len = file1.shape[0] + file4.shape[0] + file5.shape[0] + file6.shape[0]
            for i in range(total_len):
                if i < file1.shape[0]:
                    global_idx_to_local[i] = (0, i)
                    
                elif i < file1.shape[0] + file4.shape[0]:
                    global_idx_to_local[i] = (1, i - file1.shape[0])
                    
                elif i < file1.shape[0] + file4.shape[0] + file5.shape[0]:
                    global_idx_to_local[i] = (2, i - file1.shape[0] - file4.shape[0])
                    
                else:
                    global_idx_to_local[i] = (3, i - file1.shape[0] - file4.shape[0] - file5.shape[0])
                    
            data_files = [file1, file4, file5, file6]
            truth_lines = read_slice_of_file(anchor_path, 0, 450)
            truth_lines2 = read_slice_of_file(anchor_path2, 0, 2760)
            self.absolut_pos = np.concatenate(
                (
                    np.loadtxt(truth_lines).reshape(slice_num, 2),
                    np.loadtxt(truth_lines2).reshape(slice_num_r2, 2),
                ),
                axis=0,
            )
        else:
            global_idx_to_local = {}
            for i in range(file1.shape[0]):
                if i < file1.shape[0]:
                    global_idx_to_local[i] = (0, i)
            data_files = [file1]
            anchor_path = r"/Data3/cao/ZiHanCao/datasets/huawei/round3Pos123P3.txt"
            truth_lines = read_slice_of_file(anchor_path, 450, 900)
            self.absolut_pos = np.loadtxt(truth_lines).reshape(slice_num, 2)
            
        self.global_idx_to_local = global_idx_to_local        
        self.data_files = data_files

    def __len__(self):
        return int((slice_num + slice_num_r1 + slice_num_r2) * factor)
    
    def __getitem__(self, index):
        global_idx = select_set[index, 0]
        dataset_idx, idx = self.global_idx_to_local[global_idx]

        rand_data = self.data_files[dataset_idx][idx]

        return (
            rand_data,
            self.absolut_pos[global_idx: global_idx + 1, :],
        )


class TestDataset(Dataset):

    def __init__(self):
        logger.warning(
            '[red] load all files into RAM, be careful with the memory usage [/red]'
        )
        
        data_dir1 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos2/test_new64.h5"
        file1 = h5py.File(data_dir1, "r")['data'][:]
        get_virtual_memory(logger)
        
        # data_dir2 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos2/test.h5"
        # file2 = h5py.File(data_dir2, "r")['data']
        # data_dir3 = r"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos3/test.h5"
        # file3 = h5py.File(data_dir3, "r")['data']
        
        
        anchor_path = r"/Data3/cao/ZiHanCao/datasets/huawei/round3Pos123P3_test.txt"
        
        # cat file1 and file2 together
        # global idx -> (dataset idx, idx in dataset)
        global_idx_to_local = {}
        #total_len = file1.shape[0] + file2.shape[0] +file3.shape[0]
        total_len = file1.shape[0]
        for i in range(total_len):
            if i < file1.shape[0]:
                global_idx_to_local[i] = (0, i)
            # elif i <file1.shape[0]+file2.shape[0]:
            #     global_idx_to_local[i] = (1, i-file1.shape[0])
            # else:
            #     global_idx_to_local[i] = (2, i - file1.shape[0]-file2.shape[0])
        self.global_idx_to_local = global_idx_to_local
                
        #data_files = [file1, file2]
        #data_files = [file1,file2,file3]
        data_files = [file1]
        self.data_files = data_files
        
        self.datalist = []
        truth_lines = read_slice_of_file(anchor_path,50,100)
        self.absolut_pos = np.loadtxt(truth_lines).reshape(slice_num_test, 2)
            
    def __len__(self):
        return slice_num_test

    def __getitem__(self, index):
        global_idx = index
        dataset_idx, idx = self.global_idx_to_local[global_idx]

        rand_data = self.data_files[dataset_idx][idx]

        return (
            rand_data,
            self.absolut_pos[global_idx: global_idx + 1, :],
        )


def main(opt):
    logger.info("loading dataset ...")
    train_dataset = TrainDataset()
    test_dataset = TestDataset()

    # model = generate_resnet3d(10)

    if opt.fp16:
        grad_scaler = GradScaler()

    # model.train(True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    num_epoch = opt.epochs

    tbar, (iter_task, test_iter_task) = EasyProgress.easy_progress(
        ["training..." if not finetune else "finetuning...", "testing..."],
        task_total=[len(train_loader), len(test_loader)],
        tbar_kwargs={"console": logger._console},
    )

    if finetune:
        ## resnet3d
        # import sys
        # sys.path.append('/Data3/cao/ZiHanCao/huawei_contest/code3')
        # import time_resnet3d
        # state_dict = torch.load(pretrained_weight).named_parameters()
        # logger.info('load pretrained weight')
        
        # for name, param in state_dict:
        #     for name2, param2 in model.named_parameters():
        #         if name == name2:
        #             logger.info(f'found {name}')
        #             if param.shape == param2.shape:
        #                 param2.data.copy_(param.data)
        #                 break
        #             else:
        #                 logger.warning('shape is the same')
        
        model = torch.load(pretrained_weight)
        
        ## transformer3d
        # state_dict = torch.load(pretrained_weight)
        # model.load_state_dict(state_dict)
        # logger.info("load pretrained weight")
    
    if torch.cuda.is_available():
        model = model.cuda(device=opt.device)
        
    # from lion_pytorch import Lion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=1e-6, amsgrad=True
    )
    lr_scheduler = CosineAnnealingWarmRestartsReduce(
        optimizer,
        T_0=1500,
        T_mult=2,
        lr_mult=0.8,
        eta_min=5e-5,
    )
    critrtion = nn.L1Loss()
        
    ema_model = EMA(model, beta=0.995, update_every=1)
                        
    acc = {}

    logger.info("MEAN={}".format(MEAN))
    logger.info("STD={}".format(STD))

    for epoch in range(1, num_epoch):
        b = 0
        loss_epoch = 0

        if not opt.eval_only:
            tbar.start()
            for batch_idx, (data, target) in enumerate(train_loader, 1):
                model.train()
                input1 = data.cuda(device=opt.device).float()
                target = target.cuda(device=opt.device).float().squeeze()

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16 if opt.fp16 else torch.float32,
                ):
                    
                    # normalize
                    input1 = (input1 - MEAN.to(input1)) / STD.to(input1)
                    # target = (target - GT_MEAN.to(target)) / GT_STD.to(target)
                    # input1 = input1.mean(-1, keepdim=True)
                    
                    input1 = input1.permute(0, 4, 1, 2, 3)
                    # input1 = input1.mean(1,keepdim=True)

                    # two inputs
                    output1 = model(input1)

                    # loss functions
                    loss = critrtion(output1[:, 0], target[:, 0]) + critrtion(output1[:, 1], target[:, 1])

                # if opt.fp16:
                #     grad_scaler.scale(loss).backward()
                #     grad_scaler.unscale_(optimizer)
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.03)
                #     grad_scaler.step(optimizer)
                #     grad_scaler.update()
                # else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.03)
                optimizer.step()

                ema_model.update()
                lr_scheduler.step()

                loss_epoch += loss.item()

                tbar.update(
                    iter_task,
                    completed=batch_idx,
                    total=len(train_loader),
                    description=f"epoch: [{epoch} / {num_epoch}] iter: [{batch_idx} / {len(train_loader)}], loss:{loss.item():.3f}"
                    + " | lr: {:.4f}".format(lr_scheduler.get_last_lr()[0]),
                    visible=True if batch_idx != len(train_loader) else False,
                 )

                if (batch_idx + 1) % 100 == 0:
                    b += 1
                    ema_model.eval()
                    save_path = f"/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/{prefix_weight_name}/ep_{epoch}_iter{batch_idx}.pth"
                    #torch.save(model,save_path)
                    logger.info('save model to {}'.format(save_path))
                    with torch.no_grad():
                    # with torch.no_grad():
                        error = 0
                        loss_test = 0
                        for t_batch_idx, (data, target) in enumerate(test_loader, 1):
                            input1 = data.cuda(device=opt.device).float()
                            
                            # normalize
                            input1 = (input1 - MEAN.to(input1)) / STD.to(input1)
                            
                            input1 = input1.permute(0, 4, 1, 2, 3)
                            # input1 = input1.mean(1,keepdim=True)

                            target = target.cuda(device=opt.device).float().reshape(1, 2)
                            output1 = ema_model(input1)
                            
                            # scale to original
                            # output1 = output1 * GT_STD.to(output1) + GT_MEAN.to(output1)

                            loss = critrtion(output1, target.float())

                            if t_batch_idx % 1 == 0:
                                loss_test += loss.item()
                                logger.info(
                                    "batch:[{}/{}],loss:{:.3f}".format(
                                        t_batch_idx,
                                        len(test_loader),
                                        loss.item(),
                                    )
                                )
                                output, target_ = (
                                    output1.detach().cpu().numpy(),
                                    target.detach().cpu().numpy(),
                                )
                                error += np.sqrt(
                                    np.mean(
                                        (output[:, 0] - target_[:, 0]) ** 2
                                        + (output[:, 1] - target_[:, 1]) ** 2
                                    )
                                )

                                if opt.verbose_test_samples:
                                    logger.info(
                                        "truth: {}, {}\n".format(
                                            target[0, 0].item(), target[0, 1].item()
                                        )
                                        + "pred: {}, {}".format(
                                            output1[0][0].item(), output1[0][1].item()
                                        )
                                    )
                                    
                        tbar.update(
                            test_iter_task,
                            completed=t_batch_idx,
                            total=len(test_loader),
                            description=f"epoch:[{epoch}/{num_epoch}], loss:{loss.item():.3f}",
                            visible=True if t_batch_idx != len(test_loader) else False,
                        )
                        logger.info("test error: {}".format(error / t_batch_idx))
                        acc[
                            "epoch" + str(epoch) + "batch_idx" + str(batch_idx) + ":"
                        ] = error / t_batch_idx

                        # logger.info(
                        #     "model saved to {}".format(Path(save_path).relative_to(os.getcwd()))
                        # )
                        logger.info(acc)

                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                    logger.info(
                        "model saved to {}".format(Path(save_path))
                    )
                    
                    ema_params = ema_model.ema_model.state_dict()
                    torch.save(ema_params, save_path)
                    
                    logger.info("model saved")
                    logger.info("epoch total loss: {:.2f}".format((loss_epoch / batch_idx)))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    pos_n = 1
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--round_n", type=int, default=1)
    parser.add_argument("--pos_n", type=int, default=pos_n)

    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--verbose-test-samples", type=bool, default=True)
    opt = parser.parse_args()

    opt.fp16 = False
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    with catch_any_error():
        main(opt)
