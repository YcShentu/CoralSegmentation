import os
import time
import datetime
import logging
import numpy as np
from sklearn import metrics

import torch
from torch import nn
from torch.utils import data

from data_loader import SegDataLoader, ClsDataLoader
from model_generation import model_choice
from config import EvalHyperparameters as Hyperparameters
from image_saver import image_saver, image_saver_cam


def analyze_metrics(pred, gt):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    iou = np.sum(np.logical_and(pred, gt).astype(np.float)) \
          / np.sum(np.logical_or(pred, gt).astype(np.float))

    pr = metrics.precision_score(gt, pred, average='weighted')
    re = metrics.recall_score(gt, pred, average='weighted')
    f1 = metrics.f1_score(gt, pred, average='weighted')
    return iou, pr, re, f1


@torch.no_grad()
def eval(config):

    # choose dataset for different training purpose
    if config.segmentation_only:
        image_dataset = SegDataLoader(data_dir=config.data_dir,
                                      phase='eval',
                                      data_size=config.data_size)

    else:
        image_dataset = ClsDataLoader(data_dir=config.data_dir,
                                      phase='eval',
                                      data_size=config.data_size)

    print('evaluation dataset has {} images'.format(len(image_dataset)))

    dataset_loader = data.DataLoader(image_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=4)

    # set models
    model, classifier = model_choice(config.model_name,
                                     config.out_channels)

    model_file = os.path.join(config.out_dir,
                              'models',
                              str(config.model_prefix)+'.pt')

    assert os.path.exists(model_file), \
        'pretrained model file ({}) does not exist, please check'.format(model_file)

    checkpoint = torch.load(model_file, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['seg'], strict=False)
        classifier.load_state_dict(checkpoint['cls'], strict=False)
    except Exception as e:
        print(e)

    print('loading checkpoint from {}'.format(str(config.model_prefix)+'.pt'))
    print('loss: {}'.format(checkpoint['loss']))

    CELoss = nn.CrossEntropyLoss()

    print('running on {}'.format(config.device))

    model = model.to(device=config.device)
    classifier = classifier.to(device=config.device)
    model.eval()
    classifier.eval()

    for i, (images, labels) in enumerate(dataset_loader):
        start = time.time()

        images = images.to(device=config.device)
        labels = labels.to(device=config.device)

        outputs = model(images)
        outputs_mask = torch.argmax(outputs, dim=1, keepdim=False)

        if config.segmentation_only:
            loss = CELoss(outputs, labels)
        else:
            classes = classifier(outputs)
            loss = CELoss(classes, labels)

        end = time.time()

        print('*'*20)
        print('{} / {} images processing_time: {:.4f} s  loss: {:.6f}'.
              format(i, len(image_dataset), end-start, loss.item()))

        if config.segmentation_only:
            image_saver(images=images, masks=outputs_mask,
                        out_dir=os.path.join(config.out_dir, 'images'),
                        phase='eval', steps=i,
                        epoch=config.model_prefix)
        else:
            cam_weights = list(classifier.parameters())[-1].data.cpu().numpy()
            image_saver_cam(images=images,
                            heatmaps=outputs, probs=cam_weights,
                            out_dir=os.path.join(config.out_dir, 'images'),
                            phase='eval', steps=i,
                            epoch=config.model_prefix)

        iou, pr, re, f1 = analyze_metrics(outputs_mask.cpu().numpy(), labels.cpu().numpy())
        logging.info('epoch:{} step:{} images processing_time:{:.4f}s  :_:{} loss: {:.6f} iou: {:.6f} pr: {:.6f} re: {:.6f} f1: {:.6f}'.
                     format(config.model_prefix, i, end - start, 0, loss.item(), iou, pr, re, f1))


if __name__ == '__main__':
    # parameter settings
    hp = Hyperparameters()
    opt = hp.opt

    log_file = os.path.join(opt.log_dir, datetime.datetime.now().strftime("%y_%m_%d_%H%M_eval_") + opt.model_name + '.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    eval(opt)

