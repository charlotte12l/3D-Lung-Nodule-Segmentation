import __init__paths
import os
import argparse
import sys
sys.path.append('/cluster/home/it_stu95/lxylib/')

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc
import json
from tensorboardX import SummaryWriter

# select a model from densenet3d, densenet2d_notsm, densenet2d_tsm
from model.densenet2d_tsm import densenet2d
from utils.loss import dice_loss
from utils.util import mixup_data, mixup_criterion
from dataloader.dataset_quantified import ClfDataset

#device_ids = [0, 1]

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--finetune_nepoch', type=int, default=80, help='number of epochs fine tune for')
parser.add_argument('--outf', type=str, default='',  help='output folder')
parser.add_argument('--file_name', type=str, default='', help='file name')
parser.add_argument('--mixup_alpha', type=float, default=1., help="alpha for beta distribution (`None` for no mixup).")
parser.add_argument('--train_batch', type=int, default=64, help='batch that get backward')
parser.add_argument('--batch_size', type=int, default=32, help='dataset and test batch size')
parser.add_argument('--write_log', type=bool, default=True,  help='if write log')
parser.add_argument('--load_model', type=bool, default=False, help='whether load model')
parser.add_argument('--load_model_file', type=str, default='', help='load model control')
opt = parser.parse_args()
print(opt)

divide_batch = opt.train_batch // opt.batch_size


def train(epoch, model, criterion, optimizer, writer, opt, data_loader):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    for batch_idx, data in enumerate(tqdm(data_loader)):
        # get data from data loader
        voxel, label, segment_label, name= data
        #if torch.cuda.is_available():
        voxel, label, segment_label = voxel.cuda(), label.cuda(), segment_label.cuda()

        # forward the model
        if opt.mixup_alpha is None:
            output, segment_output = model(voxel)
            indiv_cross_loss = criterion(output, label)
            indiv_dice_loss = dice_loss(segment_output, segment_label)
            loss = indiv_cross_loss + 0.2 * indiv_dice_loss
        else:
            mixed_x, y_a, y_b, seg_a, seg_b, lam = mixup_data(voxel, label, segment_label,
                                                              alpha=opt.mixup_alpha, with_segment=True)
            model_ret = model(mixed_x)
            output = model_ret['clf']
            segment_output = model_ret['seg']

            # do not use the classification loss
            indiv_cross_loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            indiv_dice_loss = mixup_criterion(dice_loss, segment_output, seg_a, seg_b, lam)
            loss =  indiv_dice_loss

        loss /= divide_batch
        loss.backward()
        train_loss += loss.item()

        if (batch_idx+1) % divide_batch == 0:
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 10 == 0:
            log_tmp = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(voxel), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item())
            print('\n{}'.format(log_tmp))

    train_loss /= (len(data_loader.dataset) / opt.batch_size)
    log_tmp = 'Train Epoch: {} Loss: {:.6f}'.format(epoch, train_loss)
    print(log_tmp)

    if opt.write_log:
        with open("./log/{}.txt".format(opt.file_name), "a") as log:
            log.write('{}\n'.format(log_tmp))
        writer.add_scalar('Train_loss', train_loss, epoch)
        torch.save(model.module.state_dict(), './model_saved/{}/{}_epoch_{}_dict.pkl'.format(
            opt.file_name, opt.file_name, epoch))


def eval(epoch, model, criterion, writer, opt, data_loader):
    model.eval()
    total_cross_loss = 0
    total_dice_loss = 0
    eval_loss = 0
    correct = 0
    positive_target = np.zeros(len(data_loader.dataset))
    positive_score = np.zeros(len(data_loader.dataset))
    for batch_idx, data in enumerate(tqdm(data_loader)):
        # get data from data loader
        voxel, label, segment_label, name = data
        #if torch.cuda.is_available():
        voxel, label, segment_label = voxel.cuda(), label.cuda(), segment_label.cuda()

        # forward the model
        model_ret = model(voxel)
        output = model_ret['clf']
        segment_output = model_ret['seg']
       # output, segment_output = model(voxel)
        indiv_cross_loss = criterion(output, label)
        indiv_dice_loss = dice_loss(segment_output, segment_label)
        loss = indiv_cross_loss + 0.2 * indiv_dice_loss
        total_cross_loss += indiv_cross_loss.item()
        total_dice_loss += indiv_dice_loss.item()

        eval_loss += loss.cpu().item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        possi = F.softmax(output, dim=1)
        for i, t in enumerate(label):
            pos = opt.batch_size * batch_idx + i
            positive_target[pos] = label.data[i]
            positive_score[pos] = possi.cpu().data[i][0]

    total_cross_loss /= len(data_loader.dataset)
    total_dice_loss /= (len(data_loader.dataset) / opt.batch_size)
    eval_loss /= (len(data_loader.dataset) / opt.batch_size)

    log_tmp = 'Eval set (epoch {}): CrossEntropyLoss:{:.6f} DiceLoss:{:.6f} AverageLoss:{:.6f}, ' \
              'Accuracy: {}/{} ({:.6f}%)'.format(epoch, total_cross_loss, total_dice_loss, eval_loss,
                                                 correct, len(data_loader.dataset),
                                                 100. * float(correct) / len(data_loader.dataset))
    print(log_tmp)

    # draw the ROC curve
    fpr, tpr, thresholds = roc_curve(positive_target, positive_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    auc_tmp = 'Eval AUC = {:.8f}'.format(roc_auc)
    print(auc_tmp)

    if opt.write_log:
        writer.add_scalar('Eval_AUC', roc_auc, epoch)
        writer.add_scalar('Eval_loss', eval_loss, epoch)
        writer.add_scalar('Eval_accuracy', 100. * float(correct) / len(data_loader.dataset), epoch)
        writer.add_scalar('Eval_CrossEntropyLoss', total_cross_loss, epoch)
        writer.add_scalar('Eval_DiceLoss', total_dice_loss, epoch)

        with open("./log/{}.txt".format(opt.file_name), "a") as log:
            log.write('{}\n{}\n'.format(log_tmp, auc_tmp))
        with open("./data/{}/eval_epoch{}_fpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(fpr.tolist(), f)
        with open("./data/{}/eval_epoch{}_tpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(tpr.tolist(), f)
        with open("./data/{}/eval_epoch{}_positive_target.json".format(opt.file_name, epoch), "w") as f:
            json.dump(positive_target.tolist(), f)
        with open("./data/{}/eval_epoch{}_positive_score.json".format(opt.file_name, epoch), "w") as f:
            json.dump(positive_score.tolist(), f)


def test(epoch, model, criterion, writer, opt, data_loader):
    model.eval()
    total_cross_loss = 0
    total_dice_loss = 0
    test_loss = 0
    correct = 0
    positive_target = np.zeros(len(data_loader.dataset))
    positive_score = np.zeros(len(data_loader.dataset))
    for batch_idx, data in enumerate(tqdm(data_loader)):
        # get data from data loader
        voxel, label, segment_label, name = data
        #if torch.cuda.is_available():
        voxel, label, segment_label = voxel.cuda(), label.cuda(), segment_label.cuda()

        # forward the model
        model_ret = model(voxel)
        output = model_ret['clf']
        segment_output = model_ret['seg']
        #output, segment_output = model(voxel)
        indiv_cross_loss = criterion(output, label)
        indiv_dice_loss = dice_loss(segment_output, segment_label)
        loss = indiv_cross_loss + 0.2 * indiv_dice_loss
        total_cross_loss += indiv_cross_loss.item()
        total_dice_loss += indiv_dice_loss.item()

        test_loss += loss.cpu().item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        possi = F.softmax(output, dim=1)
        for i, t in enumerate(label):
            pos = opt.batch_size * batch_idx + i
            positive_target[pos] = label.data[i]
            positive_score[pos] = possi.cpu().data[i][0]

    total_cross_loss /= len(data_loader.dataset)
    total_dice_loss /= (len(data_loader.dataset) / opt.batch_size)
    test_loss /= (len(data_loader.dataset) / opt.batch_size)
    log_tmp = 'Test set (epoch {}): CrossEntropyLoss:{:.6f} DiceLoss:{:.6f} Average loss:{:.6f}, ' \
              'Accuracy: {}/{} ({:.6f}%)'.format(epoch, total_cross_loss, total_dice_loss, test_loss, correct,
                                                 len(data_loader.dataset),
                                                 100. * float(correct) / len(data_loader.dataset))
    print(log_tmp)

    # draw the ROC curve
    fpr, tpr, thresholds = roc_curve(positive_target, positive_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    auc_tmp = 'Test AUC = {:.8f}'.format(roc_auc)
    print(auc_tmp)

    if opt.write_log:
        writer.add_scalar('Test_AUC', roc_auc, epoch)
        writer.add_scalar('Test_loss', test_loss, epoch)
        writer.add_scalar('Test_accuracy', 100. * float(correct) / len(data_loader.dataset), epoch)
        writer.add_scalar('Test_CrossEntropyLoss', total_cross_loss, epoch)
        writer.add_scalar('Test_DiceLoss', total_dice_loss, epoch)

        with open("./log/{}.txt".format(opt.file_name), "a") as log:
            log.write('{}\n{}\n'.format(log_tmp, auc_tmp))
        with open("./data/{}/test_epoch{}_fpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(fpr.tolist(), f)
        with open("./data/{}/test_epoch{}_tpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(tpr.tolist(), f)
        with open("./data/{}/test_epoch{}_positive_target.json".format(opt.file_name, epoch), "w") as f:
            json.dump(positive_target.tolist(), f)
        with open("./data/{}/test_epoch{}_positive_score.json".format(opt.file_name, epoch), "w") as f:
            json.dump(positive_score.tolist(), f)


def main():
    for eval_set in range(1, 6):
        opt.file_name = 'densesharp_not_patient_3_8_4_batch_crossval_{}_batch_{}'.format(eval_set, opt.train_batch)
        if opt.write_log:
            writer = SummaryWriter('./runs/{}_epoch_{}'.format(opt.file_name, opt.nepoch))
            if not os.path.exists('./data/{}'.format(opt.file_name)):
                os.makedirs('./data/{}'.format(opt.file_name))

            if not os.path.exists('./model_saved/{}'.format(opt.file_name)):
                os.makedirs('./model_saved/{}'.format(opt.file_name))
        else:
            writer = None

        train_subset = [1, 2, 3, 4, 5]
        train_subset.remove(eval_set)
        train_dataset = ClfDataset(train=True, crop_size=32, move=5, subset=train_subset, lidc=True, patient=False,
                                   output_segment=True)
        train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=None)

        eval_dataset = ClfDataset(train=False, crop_size=32, subset=train_subset, lidc=True, patient=False,
                                  output_segment=True)
        eval_data_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False, sampler=None)

        test_dataset = ClfDataset(train=False, crop_size=32, subset=[eval_set], lidc=True, patient=False,
                                  output_segment=True)
        test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler=None)

        # define models
        pkl_step_list = [145, 149, 123, 173, 190]
        pkl_name = '{}_epoch_{}_dict.pkl'.format(opt.file_name, pkl_step_list[eval_set - 1])
        opt.load_model_file = './model_saved/{}/{}'.format(opt.file_name, pkl_name)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        model = densenet2d(with_segment=True)


        #if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        print('----- Start loading Densesharp! -----')
        if opt.load_model:
            model.load_state_dict(torch.load(opt.load_model_file))
            print('-----------Load succeed!------------')
        else:
            print('------------Load failed!------------')

        model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters())
        finetune_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        milestones = [1, 20, 40, 60]
        scheduler = MultiStepLR(finetune_optimizer, milestones=milestones, gamma=0.5)

        if not opt.load_model:
            for epoch in range(opt.nepoch):
                train(epoch, model, criterion, optimizer, writer, opt, train_data_loader)##
                eval(epoch, model, criterion, writer, opt,  eval_data_loader)
                test(epoch, model, criterion, writer, opt,  test_data_loader)
        else:
            scheduler.step()
            for epoch in range(opt.finetune_nepoch):
                scheduler.step()
                train(epoch+opt.nepoch, model, criterion, finetune_optimizer, writer, opt, train_data_loader)
                eval(epoch+opt.nepoch, model, criterion, writer, opt, eval_data_loader)
                test(epoch+opt.nepoch, model, criterion, writer, opt, test_data_loader)

        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
