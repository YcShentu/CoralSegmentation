import os
import time
import datetime
import logging

import torch
from torch.utils import data

from data_loader import DataLoaderInference
from model_generation import model_choice
from config import InferenceParameters
from image_saver import image_saver_inference, image_saver_cam_inference


@torch.no_grad()
def inference(config):
    # loading data
    image_dataset = DataLoaderInference(data_dir=config.data_dir,
                                        data_size=config.data_size)

    logging.info('There are {} images'.format(len(image_dataset)))

    dataset_loader = data.DataLoader(image_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)

    # set models
    model, classifier = model_choice(config.model_name,
                                     config.out_channels)

    model_file = os.path.join(config.model_dir)

    assert os.path.exists(model_file), \
        logging.error('pretrained model file ({}) does not exist, please check'.format(model_file))
    if config.model_prefix <= 0:
        logging.warning('The loaded model is trained with {} epochs'.format(config.model_prefix))

    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['seg'], strict=False)
    classifier.load_state_dict(checkpoint['cls'], strict=False)

    logging.info('loading checkpoint from {}'.format(str(config.model_prefix)+'.pt'))
    logging.info('loss: {}'.format(checkpoint['loss']))
    logging.info('running on {}'.format(config.device))

    model = model.to(device=config.device)
    classifier = classifier.to(device=config.device)
    model.eval()
    classifier.eval()

    for i, (filenames, images) in enumerate(dataset_loader):
        try:
            start = time.time()

            images = images.to(device=config.device)

            outputs = model(images)
            outputs_mask = torch.argmax(outputs, dim=1, keepdim=False)

            end = time.time()

            logging.info('*'*20)
            logging.info('{} / {}, processing_time: {:.4f} s, {}'.
                         format(i, len(image_dataset), end-start, filenames[0]))

            if config.segmentation_only:
                image_saver_inference(images=images,
                                      masks=outputs_mask,
                                      out_dir=config.out_dir,
                                      filenames=filenames,
                                      epoch=config.model_prefix)
            else:
                cam_weights = list(classifier.parameters())[-1].data.cpu().numpy()
                image_saver_cam_inference(images=images,
                                          heatmaps=outputs,
                                          probs=cam_weights,
                                          out_dir=config.out_dir,
                                          filenames=filenames,
                                          epoch=config.model_prefix)
        except Exception as e:
            logging.error(e)


if __name__ == '__main__':
    # parameter settings
    hp = InferenceParameters()
    opt = hp.opt

    log_file = os.path.join(opt.log_dir, datetime.datetime.now().strftime("%y_%m_%d_%H%M_inference_") + opt.model_name + '.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    inference(opt)
