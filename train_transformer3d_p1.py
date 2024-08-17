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
from accelerate import Accelerator

from Transformer3D import used_model
from task_utils import EasyProgress, easy_logger, catch_any_error, getMemInfo
from utilities import CosineAnnealingWarmRestartsReduce
from constants import TrainingConstants

# logger
logger = easy_logger()
const = TrainingConstants()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    posN = 1
    parser.add_argument("--numEpochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--roundN", type=int, default=1)
    parser.add_argument("--posN", type=int, default=posN)
    parser.add_argument("--emaBeta", type=int, default=0.995)

    parser.add_argument("--printTestDistances", type=bool, default=True)
    parser.add_argument('--valPerIter', type=int, default=200)
    
    # training
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--numWorkers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--pretrainedWeightPath", type=str, default=None)
    parser.add_argument("--prefixedWeightDirName", type=str, default="ckpts/")
    parser.add_argument("--prefixWeightName", type=str, default="TransformerP1")
    
    # dataset files
    parser.add_argument("--R3DataPath", type=str, default="h5files/data/Round3Pos1/train_new64.h5")
    parser.add_argument("--R2DataPathP1", type=str, default="h5files/data/Round2Pos1/train.h5")
    parser.add_argument("--R2DataPathP2", type=str, default="h5files/data/Round2Pos2/train.h5")
    parser.add_argument("--R2DataPathP3", type=str, default="h5files/data/Round2Pos3/train.h5")
    parser.add_argument("--R3GTPath", type=str, default="h5files/anchor/round3Pos123P3.txt")
    parser.add_argument("--R2GTPath", type=str, default="h5files/anchor/round2Pos123P3.txt")
    parser.add_argument("--R3TestDataPath", type=str, default="h5files/data/test/test_new64.h5")
    parser.add_argument("--R3TestGTPath", type=str, default="h5files/anchor/round3Pos123P3_test.txt")
    
    opt = parser.parse_args()
    
    if opt.finetune:
        assert opt.pretrainedWeightPath != "", "pretrained_weight should not be empty"
        opt.prefixWeightName = "finetune" + opt.prefixWeightName
    
    finetune = opt.finetune
    # some default values
    factor = const.trainFactor[opt.posN]
    ftFactor = const.ftFactor[opt.posN]
    
    opt.factor = factor if not finetune else ftFactor
    opt.R3N = const.R3NTrain  # 500 total, split 450 for training, 50 for testing
    opt.R2N = const.R2NTrain if not finetune else 0
    opt.R3TestN = const.R3NValid
        
    # fp16
    opt.fp16 = False
    return opt

# get options for training
opt = parse_opt()

def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines

######################################################## MAIN ########################################################

def prepare_matrix_set(R3N, R2N, f):
    logger.info("preparing select set ...")
    select_set = np.zeros((int((R3N + R2N) * f), 1)).astype(int)
    cnt = 0
    for i in track(
        range(int((R3N + R2N) * f)),
        total=int((R3N + R2N) * f),
    ):
        select_set[cnt, 0] = np.random.randint(0, R3N + R2N)
        cnt += 1
    
    return select_set

# prepare some global variables
logger.info('preparing indices ...')
indices = prepare_matrix_set(opt.R3N, opt.R2N, opt.factore)
fact = opt.factor
TrainingNR3 = opt.R3N
TrainingNR2 = opt.R2N
R3TestN = opt.R3TestN

class FirstTrainOrFinetuneDataset(Dataset):
    """
    Dataset class for training or fine-tuning.
    Args:
        opt (object): Options object containing various parameters.
    Attributes:
        fact (float): Scaling factor.
        TrainingNR3 (int): Number of training samples for R3.
        TrainingNR2 (int): Number of training samples for R2.
        R3TestN (int): Number of test samples for R3.
        GT (ndarray): Ground truth data.
        h5FileIdxMapping (dict): Mapping of indices to h5 file indices.
        dataF (list): List of data files.
    Methods:
        __init__(self, opt): Initializes the dataset.
        trainIdxMapping(self, file1, file2, file3, file4): Creates the mapping of indices to h5 file indices.
        __len__(self): Returns the length of the dataset.
        __getitem__(self, index): Returns a specific item from the dataset.
    """
    
    """
    Creates the mapping of indices to h5 file indices.
    Args:
        file1 (ndarray): Data from file 1.
        file2 (ndarray): Data from file 2.
        file3 (ndarray): Data from file 3.
        file4 (ndarray): Data from file 4.
    Returns:
        dict: Mapping of indices to h5 file indices.
    """
    """
    Returns the length of the dataset.
    Returns:
        int: Length of the dataset.
    """
    """
    Returns a specific item from the dataset.
    Args:
        index (int): Index of the item to retrieve.
    Returns:
        tuple: Tuple containing the item and its ground truth data.
    """
    
    def __init__(self, opt):
        super().__init__()
        self.fact = opt.factor
        self.TrainingNR3 = opt.R3N
        self.TrainingNR2 = opt.R2N
        self.R3TestN = opt.R3TestN
        
        R3File = h5py.File(opt.R3DataPath, "r")["data"][:]
        getMemInfo(logger)

        if not opt.finetune:
            R2FileP1 = h5py.File(opt.R2DataPathP1, "r")["data"][:]
            getMemInfo(logger)

            R2FileP2 = h5py.File(opt.R2DataPathP2, "r")["data"][:]
            getMemInfo(logger)

            R2FileP3 = h5py.File(opt.R2DataPathP3, "r")["data"][:]
            getMemInfo(logger)

            acPath = opt.R3GTPath
            acPath2 = opt.R2GTPath

            h5FileIdxMapping = self.trainIdxMapping(R3File, R2FileP1, R2FileP2, R2FileP3)

            dataFiles = [R3File, R2FileP1, R2FileP2, R2FileP3]
            tL = read_slice_of_file(acPath, 0, opt.R3N)
            tL_n = read_slice_of_file(acPath2, 0, opt.R2N)
            self.GT = np.concatenate(
                (
                    np.loadtxt(tL).reshape(opt.R3N, 2),
                    np.loadtxt(tL_n).reshape(opt.R2N, 2),
                ),
                axis=0,
            )
        else:
            h5FileIdxMapping = {}
            for i in range(R3File.shape[0]):
                if i < R3File.shape[0]:
                    h5FileIdxMapping[i] = (0, i)
            dataFiles = [R3File]
            acPath = opt.R3GTPath
            tL = read_slice_of_file(acPath, 0, opt.R3N)
            self.GT = np.loadtxt(tL).reshape(opt.R3N, 2)

        self.h5FileIdxMapping = h5FileIdxMapping
        self.dataF = dataFiles
        
    def trainIdxMapping(self, file1, file2, file3, file4):
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
        
        return h5FileIdxMapping

    def __len__(self):
        return int((self.TrainingNR3 + self.TrainingNR2) * self.fact)

    def __getitem__(self, index):
        GIdx = indices[index, 0]
        h5Idx, idx = self.h5FileIdxMapping[GIdx]
        xx = self.dataF[h5Idx][idx]
        return (
            xx,
            self.GT[GIdx : GIdx + 1, :],
        )


class FirstTestOrFinetuneDataset(Dataset):
    """Dataset class for the first test or fine-tuning.
    Args:
        opt (object): Options object containing the necessary paths.
    Attributes:
        global_idx_to_local (dict): Mapping of global indices to local indices.
        dataF (list): List of data files.
        datalist (list): List of data items.
        GT (ndarray): Ground truth data.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data item at the given index.
        
    """
    def __init__(self, opt):
        super().__init__()
        
        R3DataPath = opt.R3TestDataPath
        file1 = h5py.File(R3DataPath, "r")["data"][:]
        getMemInfo(logger)
        
        acPath = opt.R3TestGTPath
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
        self.GT = np.loadtxt(TLs).reshape(opt.R3TestN, 2)

    def __len__(self):
        return R3TestN

    def __getitem__(self, index):
        GIdx = index
        DIdx, idx = self.global_idx_to_local[GIdx]

        xx = self.dataF[DIdx][idx]

        return (
            xx,
            self.GT[GIdx : GIdx + 1, :],
        )

     
def distanceFn(output: "torch.Tensor | np.ndarray",
               target: "torch.Tensor | np.ndarray"):
    """
    Calculate the Euclidean distance between the output and the target.

    Args:
        output (torch.Tensor | np.ndarray): The model's output, shape (N, 2).
        target (torch.Tensor | np.ndarray): The target values, shape (N, 2).

    Returns:
        torch.Tensor | float: The Euclidean distance between the output and the target.
                              If the input is a torch.Tensor, returns a torch.Tensor;
                              if the input is a np.ndarray, returns a float.
    """   
    if torch.is_tensor(output):
        return torch.sqrt(
            torch.mean(
                (output[:, 0] - target[:, 0]) ** 2 + \
                (output[:, 1] - target[:, 1]) ** 2
            )
        )
    elif isinstance(output, np.ndarray):
        return np.sqrt(np.mean(
                    (output[:, 0] - target[:, 0]) ** 2 + \
                    (output[:, 1] - target[:, 1]) ** 2
                ))

def main(opt):
    # get accelerator
    accelerator = Accelerator(fp16=opt.fp16)
    
    # prepare select set
    IdxMapping = prepare_matrix_set(TrainingNR3, TrainingNR2, fact)
    
    # load dataset
    logger.info("loading dataset ...")
    train_dataset = FirstTrainOrFinetuneDataset()
    test_dataset = FirstTestOrFinetuneDataset()

    if opt.fp16:
        gradScaler = GradScaler()
        
    # get dataloader
    trainDl = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.numWorkers,
        pin_memory=True,
        prefetch_factor=opt.prefetch_factor,
    )
    valDl = DataLoader(
        test_dataset, 
        batch_size=1,   # 1 is enough
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )

    # progress bar
    tbar, (trainIterTask, testIterTask) = EasyProgress.easy_progress(
        ["training..." if not opt.finetune else "finetuning...", "testing..."],
        task_total=[len(trainDl), len(valDl)],
        tbar_kwargs={"console": logger._console},
    )
    
    # model
    net = used_model()

    # finetune loading
    if opt.finetune:
        state_dict = torch.load(opt.pretrainedWeightPath)
        net.load_state_dict(state_dict)
        logger.info("load pretrained weight")
    
    if torch.cuda.is_available():
        net = net.cuda(device=opt.device)
    
    # Lion optimizer and lr scheduler
    optimizer = Lion(
        net.parameters(),
        lr=opt.lr,
        weight_decay=1e-6,
        betas=(0.9, 0.995),
    )
    lrScheduler = CosineAnnealingWarmRestartsReduce(
        optimizer,
        T_0=2000,
        T_mult=2,
        lr_mult=0.6,
        eta_min=5e-5,
    )
    
    # prepare in accelerator
    net, optimizer, trainDl, valDl = accelerator.prepare(net, optimizer, trainDl, valDl)
    
    # loss function
    lossFn = nn.L1Loss()
        
    # get exponential moving average model
    emaModel = EMA(net, beta=opt.emaBeta, update_every=1)

    # overall accuracies
    trainingAcc = {}

    logger.info('start training...')
    tbar.start()
    
    # train
    for epoch in range(1, opt.numEpochs):
        b = 0
        lossInEp = 0
        for trainBatchIdx, (xx, gt) in enumerate(trainDl, 1):
            net.train()
            xx = xx.cuda(device=opt.device).float()
            gt = gt.cuda(device=opt.device).float().squeeze()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16 if opt.fp16 else torch.float32,
            ):
                xy = net(xx)
                loss = lossFn(xy, gt)

            optimizer.zero_grad()
            if opt.fp16:
                gradScaler.scale(loss).backward()
                gradScaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.03)
                gradScaler.step(optimizer)
                gradScaler.update()
            else:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(net.parameters(), 0.03)
                optimizer.step()

            emaModel.update()
            lrScheduler.step()
            
            lossInEp += loss.item()
            tbar.update(
                trainIterTask,
                completed=trainBatchIdx,
                total=len(trainDl),
                description=f"epoch: [{epoch} / {opt.numEpochs}] iter: [{trainBatchIdx} / {len(trainDl)}], loss:{loss.item():.3f}"
                + " | lr: {:.4f}".format(lrScheduler.get_last_lr()[0]),
                visible=True if trainBatchIdx != len(trainDl) else False,
            )

            if trainBatchIdx % opt.valPerIter == 0:
                b += 1
                net.eval()
                emaModel.eval()
                save_path = f"{opt.prefixedWeightDirName}/{opt.prefixWeightName}/Ep{epoch}Iter{trainBatchIdx}.pth"
                logger.info("save model to {}".format(save_path))
                with torch.no_grad():
                    distance = 0
                    lossVal = 0
                    for testBatchIdx, (xx, gt) in enumerate(valDl, 1):
                        xx = xx.cuda(device=opt.device).type(torch.float32)
                        gt = gt.cuda(device=opt.device).type(torch.float32).flatten(1)
                        xy = emaModel(xx)
                        
                        loss = lossFn(xy, gt.float())
                        lossVal += loss.item()
                        logger.info(
                            "batch:[{}/{}],loss:{:.3f}".format(
                                testBatchIdx,
                                len(valDl),
                                loss.item(),
                            )
                        )
                        output, gt = (
                            xy.detach().cpu().numpy(),
                            gt.detach().cpu().numpy(),
                        )
                        distance += distanceFn(output, gt)

                        if opt.printTestDistances:
                            logger.info(
                                "gt: {}, {}\n".format(
                                    gt[0, 0].item(), gt[0, 1].item()
                                )
                                + "net: {}, {}".format(
                                    xy[0][0].item(), xy[0][1].item()
                                )
                            )

                    tbar.update(
                        testIterTask,
                        completed=testBatchIdx,
                        total=len(valDl),
                        description=f"epoch:[{epoch}/{opt.numEpochs}], loss:{loss.item():.3f}",
                        visible=True if testBatchIdx != len(valDl) else False,
                    )
                    logger.info("distance: {}".format(distance / testBatchIdx))
                    trainingAcc[
                        "epoch" + str(epoch) + "batch_idx" + str(trainBatchIdx) + ":"
                    ] = (distance / testBatchIdx)

                    logger.info(trainingAcc)

                # make sure the directory exists
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                # save EMA model
                logger.info("model saved to {}".format(save_path))
                ema_params = emaModel.ema_model.state_dict()
                accelerator.save(ema_params, save_path)
                logger.info("model saved")
                logger.info(
                    "epoch total loss: {:.2f}".format((lossInEp / trainBatchIdx))
                )

if __name__ == "__main__":
    opt = parse_opt()
    with catch_any_error():
        main(opt)
