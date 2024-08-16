import itertools
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import OrderedDict
import argparse
from tqdm import tqdm
from rich.progress import track
from ema_pytorch import EMA
from lion_pytorch import Lion

from Transformer3D import used_model
from task_utils import EasyProgress, easy_logger, catch_any_error, getMemInfo
from utilities import CosineAnnealingWarmRestartsReduce

# logger
logger = easy_logger()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    pos_n = 1
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--round_n", type=int, default=1)
    parser.add_argument("--pos_n", type=int, default=pos_n)

    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--verbose_test_samples", type=bool, default=True)
    
    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--pretrained_weight", type=str, default=None)
    parser.add_argument("--prefixed_weight_path", type=str, default="ckpts/")
    parser.add_argument("--prefix_weight_name", type=str, default="TransformerP1")
    
    # dataset files
    parser.add_argument("--data_dir1", type=str, default="h5files/data/Round3Pos1/train_new64.h5")
    parser.add_argument("--data_dir2", type=str, default="h5files/data/Round2Pos1/train.h5")
    parser.add_argument("--data_dir3", type=str, default="h5files/data/Round2Pos2/train.h5")
    parser.add_argument("--data_dir4", type=str, default="h5files/data/Round2Pos3/train.h5")
    parser.add_argument("--anchor_path", type=str, default="h5files/anchor/round3Pos123P3.txt")
    parser.add_argument("--anchor_path2", type=str, default="h5files/anchor/round2Pos123P3.txt")
    parser.add_argument("--test_data_dir", type=str, default="h5files/data/test/test_new64.h5")
    parser.add_argument("--test_anchor_path", type=str, default="h5files/anchor/round3Pos123P3_test.txt")
    
    opt = parser.parse_args()
    
    if opt.finetune:
        assert opt.pretrained_weight != "", "pretrained_weight should not be empty"
        opt.prefix_weight_name = "finetune" + opt.prefix_weight_name
    
    finetune = opt.finetune
    # some default values
    opt.factor = 200 if not finetune else 100
    opt.slice_num = 450
    opt.slice_num_r2 = 2760 if not finetune else 0
    opt.slice_num_test = 50
        
    # fp16
    opt.fp16 = False
    return opt

# training MEAN and STD
MEAN = torch.tensor([-5.091097, -5.1093035, -5.0822234, -5.0828133], dtype=torch.float32)[None, None, None, None]
STD = torch.tensor([5.861487, 5.879711, 5.8663826, 5.875907], dtype=torch.float32)[None, None, None, None]

# get options for training
opt = parse_opt()

factor = opt.factore
slice_num = opt.slice_num
slice_num_r2 = opt.slice_num_r2
slice_num_test = opt.slice_num_test


def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines

######################################################## MAIN ########################################################

def prepare_matrix_set(slice_num, slice_num_r2, factor):
    logger.info("preparing select set ...")
    select_set = np.zeros((int((slice_num + slice_num_r2) * factor), 1), dtype=np.int32)
    cnt = 0
    for i in track(
        range(int((slice_num + slice_num_r2) * factor)),
        total=int((slice_num + slice_num_r2) * factor),
    ):
        select_set[cnt, 0] = np.random.randint(0, slice_num + slice_num_r2)
        cnt += 1
    
    return select_set

logger.info('preparing indices ...')
indices = prepare_matrix_set(slice_num, slice_num_r2, factor)

class FirstTrainOrFinetuneDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        
        file1 = h5py.File(opt.data_dir1, "r")["data"][:]
        getMemInfo(logger)

        if not opt.finetune:
            file2 = h5py.File(opt.data_dir2, "r")["data"][:]
            getMemInfo(logger)

            file3 = h5py.File(opt.data_dir3, "r")["data"][:]
            getMemInfo(logger)

            file4 = h5py.File(opt.data_dir4, "r")["data"][:]
            getMemInfo(logger)

            acPath = opt.anchor_path
            acPath2 = opt.anchor_path2

            h5FileIdxMapping = {}
            total_len = (
                file1.shape[0] + file2.shape[0] + file3.shape[0] + file4.shape[0]
            )
            for i in range(total_len):
                if i < file1.shape[0]:
                    h5FileIdxMapping[i] = (0, i)
                elif i < file1.shape[0] + file2.shape[0]:
                    h5FileIdxMapping[i] = (1, i - file1.shape[0])
                elif i < file1.shape[0] + file2.shape[0] + file3.shape[0]:
                    h5FileIdxMapping[i] = (2, i - file1.shape[0] - file2.shape[0])
                else:
                    h5FileIdxMapping[i] = (3, i - file1.shape[0] - file2.shape[0] - file3.shape[0])

            dataFiles = [file1, file2, file3, file4]
            tL = read_slice_of_file(acPath, 0, opt.slice_num)
            tL_n = read_slice_of_file(acPath2, 0, opt.slice_num_r2)
            self.GT = np.concatenate(
                (
                    np.loadtxt(tL).reshape(opt.slice_num, 2),
                    np.loadtxt(tL_n).reshape(opt.slice_num_r2, 2),
                ),
                axis=0,
            )
        else:
            h5FileIdxMapping = {}
            for i in range(file1.shape[0]):
                if i < file1.shape[0]:
                    h5FileIdxMapping[i] = (0, i)
            dataFiles = [file1]
            acPath = opt.anchor_path
            tL = read_slice_of_file(acPath, 0, opt.slice_num)
            self.GT = np.loadtxt(tL).reshape(opt.slice_num, 2)

        self.h5FileIdxMapping = h5FileIdxMapping
        self.dataF = dataFiles

    def __len__(self):
        return int((slice_num + slice_num_r2) * factor)

    def __getitem__(self, index):
        GIdx = indices[index, 0]
        h5Idx, idx = self.h5FileIdxMapping[GIdx]

        xx = self.dataF[h5Idx][idx]

        return (
            xx,
            self.GT[GIdx : GIdx + 1, :],
        )


class FirstTestOrFinetuneDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        
        data_dir1 = opt.test_data_dir
        file1 = h5py.File(data_dir1, "r")["data"][:]
        getMemInfo(logger)
        
        acPath = opt.test_anchor_path
        h5FileIdxMapping = {}
        TL = file1.shape[0]
        for i in range(TL):
            if i < file1.shape[0]:
                h5FileIdxMapping[i] = (0, i)
        self.global_idx_to_local = h5FileIdxMapping

        dataF = [file1]
        self.dataF = dataF

        self.datalist = []
        TLs = read_slice_of_file(acPath, 0, 50)
        self.GT = np.loadtxt(TLs).reshape(opt.slice_num_test, 2)

    def __len__(self):
        return slice_num_test

    def __getitem__(self, index):
        GIdx = index
        DIdx, idx = self.global_idx_to_local[GIdx]

        xx = self.dataF[DIdx][idx]

        return (
            xx,
            self.GT[GIdx : GIdx + 1, :],
        )

     
def dist_error(output, target):
    return np.sqrt(np.mean(
                (output[:, 0] - target[:, 0]) ** 2 + \
                (output[:, 1] - target[:, 1]) ** 2
            ))


def main(opt):
    # prepare select set
    select_set = prepare_matrix_set(slice_num, slice_num_r2, factor)
    
    # load dataset
    logger.info("loading dataset ...")
    train_dataset = FirstTrainOrFinetuneDataset()
    test_dataset = FirstTestOrFinetuneDataset()

    if opt.fp16:
        grad_scaler = GradScaler()
        
    # get dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        prefetch_factor=opt.prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,   # 1 is enough
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )

    # progress bar
    tbar, (iter_task, test_iter_task) = EasyProgress.easy_progress(
        ["training..." if not opt.finetune else "finetuning...", "testing..."],
        task_total=[len(train_loader), len(test_loader)],
        tbar_kwargs={"console": logger._console},
    )
    
    # model
    model = used_model()

    # finetune loading
    if opt.finetune:
        state_dict = torch.load(opt.pretrained_weight)
        model.load_state_dict(state_dict)
        logger.info("load pretrained weight")
    
    if torch.cuda.is_available():
        model = model.cuda(device=opt.device)
    
    # Lion optimizer and lr scheduler
    optimizer = Lion(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-6,
        betas=(0.9, 0.995),
    )
    lr_scheduler = CosineAnnealingWarmRestartsReduce(
        optimizer,
        T_0=2000,
        T_mult=2,
        lr_mult=0.6,
        eta_min=5e-5,
    )
    
    # loss function
    critrtion = nn.L1Loss()
        
    # get exponential moving average model
    emaModel = EMA(model, beta=0.995, update_every=1)

    # overall accuracies
    acc = {}

    # print MEAN and STD
    # logger.info("MEAN={}".format(MEAN))
    # logger.info("STD={}".format(STD))

    logger.info('start training...')
    for epoch in range(1, opt.num_epoch):
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
                    # input1 = (input1 - MEAN.to(input1)) / STD.to(input1)
                    # input1 = input1.permute(0, 4, 1, 2, 3)
                    output1 = model(input1)

                    # loss functions
                    loss = critrtion(output1, target)

                optimizer.zero_grad()
                if opt.fp16:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.03)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.03)
                    optimizer.step()

                emaModel.update()
                lr_scheduler.step()

                loss_epoch += loss.item()

                tbar.update(
                    iter_task,
                    completed=batch_idx,
                    total=len(train_loader),
                    description=f"epoch: [{epoch} / {opt.num_epoch}] iter: [{batch_idx} / {len(train_loader)}], loss:{loss.item():.3f}"
                    + " | lr: {:.4f}".format(lr_scheduler.get_last_lr()[0]),
                    visible=True if batch_idx != len(train_loader) else False,
                )

                if (batch_idx + 1) % 100 == 0:
                    b += 1
                    model.eval()
                    emaModel.eval()
                    save_path = f"{opt.prefixed_weight_path}/{opt.prefix_weight_name}/ep_{epoch}_iter{batch_idx}.pth"
                    logger.info("save model to {}".format(save_path))
                    with torch.no_grad():
                        error = 0
                        loss_test = 0
                        for t_batch_idx, (data, target) in enumerate(test_loader, 1):
                            input1 = data.cuda(device=opt.device).float()

                            # normalize
                            # input1 = (input1 - MEAN.to(input1)) / STD.to(input1)
                            # input1 = input1.permute(0, 4, 1, 2, 3)
                            target = target.cuda(device=opt.device).float().reshape(1, 2)
                            output1 = emaModel(input1)
                            
                            loss = critrtion(output1, target.float())

                            loss_test += loss.item()
                            logger.info(
                                "batch:[{}/{}],loss:{:.3f}".format(
                                    t_batch_idx,
                                    len(test_loader),
                                    loss.item(),
                                )
                            )
                            output, target = (
                                output1.detach().cpu().numpy(),
                                target.detach().cpu().numpy(),
                            )
                            error += dist_error(output, target)

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
                            description=f"epoch:[{epoch}/{opt.num_epoch}], loss:{loss.item():.3f}",
                            visible=True if t_batch_idx != len(test_loader) else False,
                        )
                        logger.info("test error: {}".format(error / t_batch_idx))
                        acc[
                            "epoch" + str(epoch) + "batch_idx" + str(batch_idx) + ":"
                        ] = (error / t_batch_idx)

                        logger.info(acc)

                    # make sure the directory exists
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                    # save EMA model
                    logger.info("model saved to {}".format(save_path))
                    ema_params = emaModel.ema_model.state_dict()
                    torch.save(ema_params, save_path)
                    logger.info("model saved")
                    logger.info(
                        "epoch total loss: {:.2f}".format((loss_epoch / batch_idx))
                    )

if __name__ == "__main__":
    opt = parse_opt()
    with catch_any_error():
        main(opt)
