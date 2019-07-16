import math
from  torch.optim.lr_scheduler import _LRScheduler

class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor =  pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]


class OneCycle(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, last_epoch=-1,
                    momentums = (0.85, 0.95), div_factor = 25, phase1=0.3):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.phase1_iters = int(self.N * phase1)
        self.phase2_iters = (self.N - self.phase1_iters)
        self.momentums = momentums
        self.mom_diff = momentums[1] - momentums[0]

        self.low_lrs = [opt_grp['lr']/div_factor for opt_grp in optimizer.param_groups]
        self.final_lrs = [opt_grp['lr']/(div_factor * 1e4) for opt_grp in optimizer.param_groups]
        super(OneCycle, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1

        # Going from base_lr / 25 -> base_lr
        if T <= self.phase1_iters:
            cos_anneling =  (1 + math.cos(math.pi * T / self.phase1_iters)) / 2
            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['momentum'] = self.momentums[0] + self.mom_diff * cos_anneling

            return [base_lr - (base_lr - low_lr) * cos_anneling 
                    for base_lr, low_lr in zip(self.base_lrs, self.low_lrs)]

        # Going from base_lr -> base_lr / (25e4)
        T -= self.phase1_iters
        cos_anneling =  (1 + math.cos(math.pi * T / self.phase2_iters)) / 2

        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['momentum'] = self.momentums[1] - self.mom_diff * cos_anneling
        return [final_lr + (base_lr - final_lr) * cos_anneling 
            for base_lr, final_lr in zip(self.base_lrs, self.final_lrs)]


if __name__ == "__main__":
    import torchvision
    import torch
    import matplotlib.pylab as plt

    resnet = torchvision.models.resnet34()
    params = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9
    }
    optimizer = torch.optim.SGD(params=resnet.parameters(), **params)

    epochs = 2
    iters_per_epoch = 100
    lrs = []
    mementums = []
    lr_scheduler = OneCycle(optimizer, epochs, iters_per_epoch)
    #lr_scheduler = Poly(optimizer, epochs, iters_per_epoch)

    for epoch in range(epochs):
        for i in range(iters_per_epoch):
            lr_scheduler.step(epoch=epoch)
            lr_scheduler(optimizer, i, epoch)
            lrs.append(optimizer.param_groups[0]['lr'])
            mementums.append(optimizer.param_groups[0]['momentum'])

    plt.ylabel("learning rate")
    plt.xlabel("iteration")
    plt.plot(lrs)
    plt.show()

    plt.ylabel("momentum")
    plt.xlabel("iteration")
    plt.plot(mementums)
    plt.show()

    