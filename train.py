import time
import datetime
import logging
import torch
# from apex import amp
from tools.utils import AverageMeter
from models.SpectralClustering import spectral_clustering, kmeans_clustering
import numpy as np
from torch_scatter import scatter

def Cluster_loss(correlation, y_pred):
    correlation = correlation.mean(0)
    corr = scatter(correlation, y_pred, dim=0, reduce="sum")
    corr = scatter(corr, y_pred, dim=1, reduce="sum")
    corr_x = torch.diag(corr, 0)
    corr_y = corr.sum(1) - corr_x
    ret_loss = (corr_x / corr_y).mean()
    return ret_loss

def train_cal(config, epoch, model, classifier, domain_classifier, criterion_cla, criterion_pair, 
    criterion_domain, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes,
    y_pred):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    domain_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()


    classifier.train()
    domain_classifier.train()

    end = time.time()
    model.eval()

    model.train()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids.cpu()]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        # t = time.time()
        features, corre_matrix = model(imgs, epoch = epoch)
        # print(corre_matrix.shape)
        outputs = classifier(features)

        n, p = features.shape
        if epoch >= config.TRAIN.START_EPOCH_CC:
            features_decor = torch.zeros_like(features).cuda()
            for i in torch.unique(y_pred):
                var = features[:, torch.where(y_pred == i)[0]]
                features_decor[:, torch.where(y_pred == i)[0]] = var[torch.randperm(n)]
        else:
            ind = torch.rand_like(features).argsort(dim=0)
            features_decor = torch.zeros_like(features).scatter_(0, ind, features)

        new_features = torch.cat([features, features_decor])
        domain_ids = torch.zeros(n * 2).cuda().long()
        domain_ids[:n] = 1
        pred_domain = domain_classifier(new_features.detach())
        _, domain_preds = torch.max(pred_domain.data, 1)
        _, preds = torch.max(outputs.data, 1)
        # print(pred_domain.shape,domain_ids.shape)
        # Update the domain discriminator
        domain_loss = criterion_domain(pred_domain, domain_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(domain_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                domain_loss.backward()
            optimizer_cc.step()
            cluster_loss = Cluster_loss(corre_matrix, y_pred)

        # Update the backbone
        new_pred_domain = domain_classifier(features)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        adv_loss = criterion_domain(new_pred_domain, torch.zeros(n).to(features).long())
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss +  config.LAMBDA1 * adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + config.LAMBDA2 * cluster_loss
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        domain_corrects.update(torch.sum(domain_preds == domain_ids.data).float()/domain_ids.size(0), domain_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(cluster_loss.item(), domain_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), domain_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'ClusterLoss:{clo_loss.avg:.4f} '
                  'TriLoss:{pair_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'DecorAcc:{deacc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss,
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, deacc = domain_corrects, pair_loss = batch_pair_loss))



def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))
