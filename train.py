import os
import time
import datetime
import logging

from visdom import Visdom
import torch
from torch import nn
from torch.utils import data

from data_loader import SegDataLoader, ClsDataLoader
from model_generation import model_choice
from config import TrainingHyperparameters as Hyperparameters
from image_saver import image_saver, image_saver_cam


def train(config):

    # choose dataset for different training purpose
    if config.segmentation_only:
        image_datasets = {phase: SegDataLoader(data_dir=config.data_dir,
                                               phase=phase,
                                               data_size=config.data_size)
                          for phase in ['train', 'eval']}
    else:
        image_datasets = {phase: ClsDataLoader(data_dir=config.data_dir,
                                               phase=phase,
                                               data_size=config.data_size)
                          for phase in ['train', 'eval']}

    print('loading dataset: train: {}, eval: {}'
          .format(len(image_datasets['train']),
                  len(image_datasets['eval'])))

    dataset_loaders = {'train': data.DataLoader(image_datasets['train'],
                                                batch_size=config.batch_size,
                                                shuffle=True,
                                                num_workers=4),

                       'eval': data.DataLoader(image_datasets['eval'],
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=0)
                       }

    model, classifier = model_choice(config.model_name, config.out_channels)

    if config.model_prefix > 0:
        model_file = os.path.join(config.out_dir, 'models', str(config.model_prefix)+'.pt')

        assert os.path.exists(model_file), \
            'pretrained model file ({}) does not exist, please check'.format(model_file)

        checkpoint = torch.load(model_file, map_location='cpu')
        try:
            model.load_state_dict(checkpoint['seg'], strict=False)
            opt1 = torch.optim.Adam(model.parameters(), lr=checkpoint['lr1'])
            classifier.load_state_dict(checkpoint['cls'], strict=False)
            opt2 = torch.optim.Adam(classifier.parameters(), lr=config.lr)

            print('loading checkpoint from {}'.format(str(config.model_prefix) + '.pt'))
            print('loss: {}'.format(checkpoint['loss']))
        except KeyError:
            opt2 = torch.optim.Adam(classifier.parameters(), lr=config.lr)
    else:
        opt1 = torch.optim.Adam(model.parameters(), lr=config.lr)
        opt2 = torch.optim.Adam(classifier.parameters(), lr=config.lr)

    lr_scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(opt1, gamma=0.99)
    lr_scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(opt2, gamma=0.9)

    CELoss = nn.CrossEntropyLoss()

    print('running on {}'.format(config.device))

    # set visdom
    if config.use_visdom:
        viz = Visdom()
        assert viz.check_connection()
        visline1 = viz.line(
            X=torch.Tensor([1]).cpu() * config.model_prefix,
            Y=torch.Tensor([0]).cpu(),
            win=1,
            opts=dict(xlabel='epochs',
                      ylabel='loss',
                      title='training loss',
                      )
        )
        visline2 = viz.line(
            X=torch.Tensor([1]).cpu() * config.model_prefix,
            Y=torch.Tensor([0]).cpu(),
            win=2,
            opts=dict(xlabel='epochs',
                      ylabel='loss',
                      title='evaluation loss',
                      )
        )
        visline3 = viz.line(
            X=torch.Tensor([1]).cpu() * config.model_prefix,
            Y=torch.Tensor([0]).cpu(),
            win=3,
            opts=dict(xlabel='epochs',
                      ylabel='LR',
                      title='Learning rate')
        )

    model = model.to(device=config.device)
    classifier = classifier.to(device=config.device)

    global_steps = {'train': 0, 'eval': 0}

    global min_loss
    min_loss = 1e5

    for epoch in range(config.model_prefix, config.epochs):
        lr_scheduler_1.step()
        lr_scheduler_2.step()

        for phase in ['train', 'eval']:
            running_loss = []

            if phase == 'train':
                model.train()
                classifier.train()
            else:
                model.eval()
                classifier.eval()

            for i, (images, labels) in enumerate(dataset_loaders[phase]):

                start = time.time()

                images = images.to(device=config.device)
                labels = labels.to(device=config.device)

                opt1.zero_grad()
                opt2.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(images)
                    outputs_mask = torch.argmax(outputs, dim=1, keepdim=False)

                    if config.segmentation_only:
                        loss = CELoss(outputs, labels)
                    else:
                        classes = classifier(outputs)
                        loss = CELoss(classes, labels)

                    running_loss.append(loss.item())

                    if phase == 'train':
                        loss.backward()
                        opt1.step()
                        opt2.step()

                end = time.time()

                print('*'*20)
                print('epoch: {}/{}  {}_global_steps: {}  processing_time: {:.4f} s  LR: {:.8f}'.
                      format(epoch, config.epochs, phase, global_steps[phase],
                             end-start, opt1.param_groups[0]['lr']))
                print('{} loss: {:.6}'.format(phase, loss.item()))

                if phase == 'train' and i % 10 == 0:
                    logging.info('epoch:{} steps:{} processing_time:{:.4f}s LR:{:.8f} loss:{:.6}'.
                                 format(epoch, global_steps[phase], end-start,
                                        opt1.param_groups[0]['lr'], loss.item()))
                if phase == 'eval':
                    logging.info('eval_epoch:{} steps:{} processing_time:{:.4f}s LR:{:.8f} loss:{:.6}'.
                                 format(epoch, global_steps[phase], end - start,
                                        opt1.param_groups[0]['lr'], loss.item()))

                # set visdom
                if config.use_visdom and i % 5 == 0:
                    if phase == 'train':
                        viz.line(
                            X=torch.Tensor([1]).cpu() *
                              (epoch + i * config.batch_size / len(image_datasets[phase])),
                            Y=torch.Tensor([loss.item()]).cpu(),
                            win=visline1,
                            update='append'
                        )
                        viz.line(
                            X=torch.Tensor([1]).cpu() *
                              (epoch + i * config.batch_size / len(image_datasets[phase])),
                            Y=torch.Tensor([opt1.param_groups[0]['lr']]),
                            win=visline3,
                            update='append'
                        )
                    else:
                        viz.line(
                            X=torch.Tensor([1]).cpu() *
                              (epoch + i * config.batch_size / len(image_datasets[phase])),
                            Y=torch.Tensor([loss.item()]),
                            win=visline2,
                            update='append'
                        )

                global_steps[phase] += 1

                if epoch % config.image_intervals == 0:
                    if config.segmentation_only:
                        image_saver(images=images, masks=outputs_mask,
                                    out_dir=os.path.join(config.out_dir, 'images'),
                                    phase=phase, steps=global_steps[phase],
                                    epoch=epoch)
                    else:
                        cam_weights = list(classifier.parameters())[-1].data.cpu().numpy()
                        image_saver_cam(images=images,
                                        heatmaps=outputs, probs=cam_weights,
                                        out_dir=os.path.join(config.out_dir, 'images'),
                                        phase=phase, steps=global_steps[phase],
                                        epoch=epoch)

            current_loss = sum(running_loss)/len(running_loss)

            if phase == 'train' and epoch % config.model_intervals == 0 and current_loss < min_loss:
                torch.save({
                    'epoch': epoch,
                    'seg': model.state_dict(),
                    'cls': classifier.state_dict(),
                    'lr1': opt1.param_groups[0]['lr'],
                    'lr2': opt2.param_groups[0]['lr'],
                    'loss': current_loss},
                     os.path.join(config.out_dir, 'models', str(epoch)+'.pt')
                )
                running_loss.clear()
                min_loss = current_loss

                print('Saving model in {} for {} epoches'.format(config.out_dir +'/models', epoch))


if __name__ == '__main__':
    # parameter settings
    hp = Hyperparameters()
    opt = hp.opt

    log_file = os.path.join(opt.log_dir, datetime.datetime.now().strftime("%y_%m_%d_%H%M_") + opt.model_name + '.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    train(opt)
