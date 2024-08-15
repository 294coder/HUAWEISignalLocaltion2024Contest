import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


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
