import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import get_data
from src.model import InterpretableMultimodalCapsuleFusion
from src.eval_metrics import *


def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


def train(train_loader, model, criterion, optimizer, args):
    results = []
    truths = []
    model.train()
    total_loss = 0.0
    total_batch_size = 0

    for ind, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        # measure data loading time
        sample_ind, text, audio, video = batch_X
        text, audio, video = text.cuda(), audio.cuda(), video.cuda()
        batch_Y = batch_Y.cuda()
        eval_attr = batch_Y.squeeze(-1)
        batch_size = text.size(0)
        total_batch_size += batch_size

        # compute output
        preds = model(text, audio, video, batch_size)
        if args.dataset == "iemocap":
            preds = preds.reshape(-1, 2)
            if args.emotion == 'neutral':     # get corresponding label
                eval_attr = eval_attr[:, 0:1]
            elif args.emotion == 'happy':
                eval_attr = eval_attr[:, 1:2]
            elif args.emotion == 'sad':
                eval_attr = eval_attr[:, 2:3]
            elif args.emotion == 'angry':
                eval_attr = eval_attr[:, 3:4]
            eval_attr = eval_attr.long().reshape(-1)
            raw_loss = criterion(preds, eval_attr)
            results.append(preds)
            truths.append(eval_attr)
        elif args.dataset in ["mosei_senti", "mosi"]:
            preds = preds.reshape(-1)
            eval_attr = eval_attr.reshape(-1)
            raw_loss = criterion(preds, eval_attr)
            results.append(preds)
            truths.append(eval_attr)

        total_loss += raw_loss.item() * batch_size
        combined_loss = raw_loss
        optimizer.zero_grad()
        combined_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def validate(loader, model, criterion, args):
    model.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    with torch.no_grad():
        for ind, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, text, audio, video = batch_X
            text, audio, video = text.cuda(), audio.cuda(), video.cuda()
            batch_Y = batch_Y.cuda()
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            batch_size = text.size(0)
            total_batch_size += batch_size
            preds = model(text, audio, video, batch_size)
            if args.dataset == 'iemocap':
                preds = preds.reshape(-1, 2)
                if args.emotion == 'neutral':
                    eval_attr = eval_attr[:, 0:1]
                elif args.emotion == 'happy':
                    eval_attr = eval_attr[:, 1:2]
                elif args.emotion == 'sad':
                    eval_attr = eval_attr[:, 2:3]
                elif args.emotion == 'angry':
                    eval_attr = eval_attr[:, 3:4]
                eval_attr = eval_attr.long().reshape(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size
            elif args.dataset in ['mosi', "mosei_senti"]:
                preds = preds.reshape(-1)
                eval_attr = eval_attr.reshape(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size
    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def adjust_learning_rate(optimizer, epoch, args, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Capsule Learner')
    parser.add_argument('--aligned', default=True, help='consider aligned experiment or not')
    parser.add_argument('--dataset', type=str, default='mosei_senti', help='dataset to use')
    parser.add_argument('--emotion', type=str, default='neutral',
                        help='we train unique model for each emotion in iemocap dataset')
    parser.add_argument('--data-path', type=str, default='../CMU-Multimodal/data', help='path for storing the dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    # parser.add_argument('-b', '--batch_size', default=32, type=int)
    # parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    # parser.add_argument('--dim_capsule', default=64, type=int, help='dimension of capsule')
    # parser.add_argument('--routing', default=4, type=int, help='total routing rounds')
    # parser.add_argument('--dropout', default=0.7)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
    parser.add_argument('--patience', default=5, type=int, help='patience for learning rate decay')
    parser.add_argument('--device', type=str, default='0', help='gpu')
    args = parser.parse_args()

    hyperparams = dict()
    hyperparams['batch_size'] = [16, 32, 64, 128]
    hyperparams['learning_rate'] = [0.01, 0.001, 0.0001]
    # hyperparams['optimizer'] = ['adam', 'RMSprop', 'momentum', 'Adagrad']
    hyperparams['optimizer'] = [args.optimizer]
    hyperparams['dim_capsule'] = [16, 32, 64]
    hyperparams['routing'] = [2, 3, 4]
    hyperparams['dropout'] = [0.3, 0.5, 0.7]

    kk_acc = 0
    kk_acc7 = 0
    kk_mae = 0
    hyperparams_best = dict()    # store the best setting
    params_acc = dict()
    params_acc7 = dict()
    params_mae = dict()

    assert args.dataset in ['mosi', 'mosei_senti', 'iemocap'], "supported datasets are mosei_senti and iemocap"

    total_setting = total(hyperparams)
    seen_setting = set()
    kk = 0

    if args.dataset == "mosei_senti":
        criterion = nn.L1Loss().cuda()
        t_in = 300
        a_in = 74
        v_in = 35
        label_dim = 1

    elif args.dataset == "iemocap":
        assert args.emotion in ['neutral', 'happy', 'sad', 'angry'], \
            "emotions in iemocap are neutral, happy, sad and angry"
        criterion = nn.CrossEntropyLoss().cuda()
        t_in = 300
        a_in = 74
        v_in = 35
        label_dim = 2

    elif args.dataset == 'mosi':
        criterion = nn.L1Loss().cuda()
        t_in = 300
        a_in = 5
        v_in = 20
        label_dim = 1

    if args.dataset in ['mosi', 'mosei_senti']:
        best_acc_all_setting = 0
        mae_best_acc_all_setting = 2
        mult_a5_best_acc_all_setting = 0
        mult_a7_best_acc_all_setting = 0
        corr_best_acc_all_setting = 0
        fscore_best_acc_all_setting = 0

        best_mae_all_setting = 2
        acc_best_mae_all_setting = 0
        mult_a5_best_mae_all_setting = 0
        mult_a7_best_mae_all_setting = 0
        corr_best_mae_all_setting = 0
        fscore_best_mae_all_setting = 0

        best_acc7_all_setting = 0
        acc_best_acc7_all_setting = 0
        mae_best_acc7_all_setting = 2
        mult_a5_best_acc7_all_setting = 0
        corr_best_acc7_all_setting = 0
        fscore_best_acc7_all_setting = 0

    elif args.dataset == 'iemocap':
        best_acc_all_setting = 0

    for i in range(1000000):
        print('there are {} setting. {}th setting now'.format(total_setting, kk+1))
        if kk >= total_setting:
            break
        batch_size = random.choice((hyperparams['batch_size']))
        lr = random.choice((hyperparams['learning_rate']))
        otm = random.choice(hyperparams['optimizer'])
        dim_capsule = random.choice(hyperparams['dim_capsule'])
        routing = random.choice(hyperparams['routing'])
        dropout = random.choice(hyperparams['dropout'])
        current_setting = (batch_size, lr, otm, dim_capsule, routing, dropout)

        hyperparams_best['batch_size'] = batch_size
        hyperparams_best['lr'] = lr
        hyperparams_best['optimizer'] = otm
        hyperparams_best['dim_capsule'] = dim_capsule
        hyperparams_best['routing'] = routing
        hyperparams_best['dropout'] = dropout

        model = InterpretableMultimodalCapsuleFusion(args,
                                                     hyperparams_best,
                                                     label_dim, t_in, a_in, v_in).cuda()
        if current_setting in seen_setting:
            continue
        else:
            seen_setting.add(current_setting)
            kk += 1

        if otm == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr)
        elif otm == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr)
        elif otm == 'momentum':
            optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.8)
        elif otm == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.1, verbose=True)

        train_data = get_data(args, args.dataset, 'train')
        valid_data = get_data(args, args.dataset, 'valid')
        test_data = get_data(args, args.dataset, 'test')

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        if args.dataset in ['mosei_senti', 'mosi']:
            best_model_acc = -1
            mae_best_model = 2
            mult_a7_best_model = -1
            mult_a5_best_model = -1
            corr_best_model = 0
            fscore_best_model = 0
            patience_valid = 0
        elif args.dataset == "iemocap":
            best_acc = 0
            best_f1 = 0
            patience_valid = 0

        best_valid_loss = 2
        # train
        for epoch in range(args.epochs):
            start = time.time()
            adjust_learning_rate(optimizer, epoch, args, lr)
            # train for one epoch
            train_loss, train_results, train_truth = train(train_loader, model, criterion, optimizer, args)
            # validate for one epoch
            valid_loss, valid_results, valid_truth = validate(valid_loader, model, criterion, args)
            test_loss, _, _ = validate(test_loader, model, criterion, args)
            end = time.time()
            duration = end - start
            print('-'*50)
            print('epoch {:2d} | time{:5.4f} sec | train loss {:5.4f} | valid loss {:5.4f} | test loss {:5.4f}'
                  .format(epoch, duration, train_loss, valid_loss, test_loss))
            print('-'*50)
            if valid_loss < best_valid_loss:
                print('find better model. Saving to ./pretrained_models/{}_best_model_{}_{}thSetting.pkl'
                      .format(args.dataset, args.device, kk))
                best_valid_loss = valid_loss
                torch.save(model, './pretrained_models/{}_best_model_{}_{}thSetting.pkl'
                           .format(args.dataset, args.device, kk))
                patience_valid = 0
            else:
                patience_valid += 1
                if patience_valid == 10:
                    break
            scheduler.step(valid_loss)

        # test
        model = torch.load('./pretrained_models/{}_best_model_{}_{}thSetting.pkl'
                           .format(args.dataset, args.device, kk))
        _, test_results, test_truth = validate(test_loader, model, criterion, args)
        if args.dataset in ["mosi", "mosei_senti"]:
            mae, corr, mult_a7, mult_a5, f_score, acc = eval_mosei_senti_or_mosi(test_results, test_truth)
            print('current setting is', hyperparams_best)
            print('acc {:5.4f} | acc7 {:5.4f} | fscore {:5.4f}'.format(acc, mult_a7, f_score))
            print('mae {:5.4f} | corr {:5.4f} | acc5 {:5.4f}'.format(mae, corr, mult_a5))
            print('-'*50)
            if best_acc_all_setting < acc:
                kk_acc = kk
                params_acc = hyperparams_best
                best_acc_all_setting = acc
                mae_best_acc_all_setting = mae
                mult_a7_best_acc_all_setting = mult_a7
                mult_a5_best_acc_all_setting = mult_a5
                corr_best_acc_all_setting = corr
                fscore_best_acc_all_setting = f_score

            if best_acc7_all_setting < mult_a7:
                kk_acc7 = kk
                params_acc7 = hyperparams_best
                best_acc7_all_setting = mult_a7
                acc_best_acc7_all_setting = acc
                mae_best_acc7_all_setting = mae
                mult_a5_best_acc7_all_setting = mult_a5
                corr_best_acc7_all_setting = corr
                fscore_best_acc7_all_setting = f_score

            if best_mae_all_setting > mae:
                kk_mae = kk
                params_mae = hyperparams_best
                best_mae_all_setting = mae
                acc_best_mae_all_setting = acc
                mult_a7_best_mae_all_setting = mult_a7
                mult_a5_best_mae_all_setting = mult_a5
                corr_best_mae_all_setting = corr
                fscore_best_mae_all_setting = f_score

            print('{}th setting is the best-acc so far'.format(kk_acc), params_acc)
            print('acc {:5.4f} | acc7 {:5.4f} | fscore {:5.4f}'
                  .format(best_acc_all_setting, mult_a7_best_acc_all_setting, fscore_best_acc_all_setting))
            print('mae {:5.4f} | corr {:5.4f} | acc5 {:5.4f}'
                  .format(mae_best_acc_all_setting, corr_best_acc_all_setting, mult_a5_best_acc_all_setting))
            print('-'*50)

            print('{}th setting is the best-acc7 so far'.format(kk_acc7), params_acc7)
            print('acc {:5.4f} | acc7 {:5.4f} | fscore {:5.4f}'
                  .format(acc_best_acc7_all_setting, best_acc7_all_setting, fscore_best_acc7_all_setting))
            print('mae {:5.4f} | corr {:5.4f} | acc5 {:5.4f}'
                  .format(mae_best_acc7_all_setting, corr_best_acc7_all_setting, mult_a5_best_acc7_all_setting))
            print('-'*50)

            print('{}th setting is the best-mae so far'.format(kk_mae), params_mae)
            print('acc {:5.4f} | acc7 {:5.4f} | fscore {:5.4f}'
                  .format(acc_best_mae_all_setting, mult_a7_best_mae_all_setting, fscore_best_mae_all_setting))
            print('mae {:5.4f} | corr {:5.4f} | acc5 {:5.4f}'
                  .format(best_mae_all_setting, corr_best_mae_all_setting, mult_a5_best_mae_all_setting))

        elif args.dataset == 'iemocap':
            acc, f1 = eval_iemocap(test_results, test_truth)
            print('current setting is', hyperparams_best)
            print('{} | acc {:5.4f} | fscore {:5.4f}'.format(args.emotion, acc, f1))
            if best_acc_all_setting < acc:
                kk_acc = kk
                best_acc_all_setting = acc
                f1_best_acc_all_setting = f1
                params_acc = hyperparams_best
            print('{}th setting is the best so far'.format(kk_acc), params_acc)
            print('{} | acc {:5.4f} | fscore {:5.4f}'.format(args.emotion, best_acc_all_setting, f1_best_acc_all_setting))



