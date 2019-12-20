import torch
import numpy as np
import time


'''transform the segment with batch to n_segment of the positions of the pixels
   input is Bx1xXxYxZ (channel is 1), output is 5-dim list, with first dim batch from 0 - (batch_size - 1)'''


def segment2n_segment(segment, n_sat, dims=5):
    # get the parameters
    segment = segment.long()
    batch_size = segment.shape[0]
    n_indiv = torch.sum(segment, dim=[1, 2, 3, 4])

    # get the indice
    segment_np = segment.cpu().numpy()
    n_segment = list(np.where(segment_np))
    if segment.is_cuda:
        for dim in range(dims):
            n_segment[dim] = torch.Tensor(n_segment[dim]).cuda().long()
    else:
        for dim in range(dims):
            n_segment[dim] = torch.Tensor(n_segment[dim]).long()

    n_segment_output = []
    for batch in range(batch_size):
        # get the set of each batch
        if n_indiv[batch] >= n_sat:
            # random sample
            select = torch.randperm(int(n_indiv[batch]))[:n_sat]
            for dim in range(dims):
                if batch == 0:
                    n_segment_output.append(n_segment[dim][n_segment[0] == batch][select])
                else:
                    n_segment_output[dim] = torch.cat((n_segment_output[dim],
                                                       n_segment[dim][n_segment[0] == batch][select]), dim=0)
        elif n_indiv[batch] > 0:
            # repeat and random sample
            quotient = n_sat // n_indiv[batch].item()
            remainder = n_sat % n_indiv[batch].item()
            select = torch.randperm(int(n_indiv[batch]))[:remainder]
            for dim in range(dims):
                n_segment_repeat = n_segment[dim][n_segment[0] == batch].repeat(quotient)
                n_segment_remider = n_segment[dim][n_segment[0] == batch][select]
                if batch == 0:
                    n_segment_output.append(torch.cat((n_segment_repeat, n_segment_remider), dim=0))
                else:
                    n_segment_output[dim] = torch.cat((n_segment_output[dim], n_segment_repeat,
                                                       n_segment_remider), dim=0)
        else:
            # if n is zero
            for dim in range(dims):
                if batch == 0:
                    n_segment_output.append(n_segment[dim][n_segment[0] == batch][select])
                else:
                    n_segment_output[dim] = torch.cat((n_segment_output[dim],
                                                       n_segment[dim][n_segment[0] == batch][select]), dim=0)
    return n_segment_output


def get_feature_segment_past(segment, segment_possi, least_select, max_select_n=1024):
    batch_size = segment.shape[0]
    segment_output = torch.zeros_like(segment)
    n_select = segment.sum([1, 2, 3, 4]).int().cpu().detach()
    n_select = torch.max(n_select, n_select.new_full((n_select.shape[0],), least_select)).numpy()
    batch_possi = segment_possi.view(segment_possi.shape[0], -1).detach().cpu().numpy()

    logits = segment_possi.exp()
    top_logits, top_index = logits.topk(max_select_n)
    top_proba = top_logits.softmax(-1)

    select = np.zeros_like(batch_possi)
    for batch in range(batch_size):
        select_array = np.random.choice(batch_possi[batch].shape[0], size=n_select[batch], replace=False,
                                        p=batch_possi[batch])
        select_array = top_index[batch, select_array].data.cpu().numpy()
        select[batch, select_array] = 1
        if segment.is_cuda:
            segment_output[batch] = torch.from_numpy(select.reshape(segment[batch].shape)).cuda()
        else:
            segment_output[batch] = torch.from_numpy(select.reshape(segment[batch].shape))
    return segment_output


def get_feature_segment(segment, least_select=10):
    batch_size = segment.shape[0]
    segment_output = torch.zeros_like(segment)
    n_select = segment.sum([1, 2, 3, 4]).int()
    select = (n_select >= least_select)

    segment_select = torch.zeros_like(segment.view(batch_size, -1))
    for batch in range(batch_size):
        if select[batch]:
            values, index = segment[batch].view(-1).topk(n_select[batch].item())
            segment_select[batch, index] = 1
            segment_output[batch] = segment_select[batch].view(segment[batch].shape)
        else:
            segment_output[batch] = torch.ones_like(segment[batch])
    return segment_output, select


def lengths2masks(lengths, max_length=None):
    if max_length is None:
        max_length = max(lengths)
    batch_size = len(lengths)
    if lengths.is_cuda:
        masks = torch.arange(max_length).repeat(batch_size, 1).cuda()
    else:
        masks = torch.arange(max_length).repeat(batch_size, 1)
    masks = masks < lengths.long().unsqueeze_(1)
    return masks


def mixup_data(x, y, seg=None, alpha=1.0, with_segment=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if x.is_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    if with_segment:
        seg_a, seg_b = seg, seg[index]
        return mixed_x, y_a, y_b, seg_a, seg_b, lam

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


if __name__ == '__main__':
    input = torch.rand(16, 1, 32, 32, 32)
    input = input > 0.99
    start = time.clock()
    output = segment2n_segment(input, 1024)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    output = output

