import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def soft_soft_max(input, temperature=2, dim=1):
    output = torch.zeros_like(input)

    if dim == 0:
        logit = torch.div(input, temperature)
        for i in range(len(logit)):
            output[i] = torch.div(logit[i].exp(), logit.exp().sum())
    elif dim == 1:
        for n, logit in enumerate(input):
            output[n] = soft_soft_max(input[n], temperature, dim=0)
    return output


class SoftLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super(SoftLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target, target_2_index=None):
        assert len(input) == len(target)
        target = target.float()
        log_soft_max = F.softmax(input, dim=1).log()
        loss = torch.zeros_like(target[0])
        for i in range(len(target)):
            loss_tmp = log_soft_max[i][0] * (1 - target[i]) + log_soft_max[i][1] * target[i]
            if not target_2_index or i not in target_2_index:
                loss.add_(loss_tmp.neg())
            else:
                loss.add_((loss_tmp * self.alpha).neg())
        return loss


def main():
    criterion = SoftLoss()
    input = torch.Tensor([[1, 100], [100, 2]])
    target = torch.Tensor([0.1, 0.001])
    loss = criterion(input, target)
    loss.backward()
    print(loss)


if __name__ == '__main__':
    main()
