import argparse
import os
import torch


class BaseHyperparameters:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def initalize(self):
        # set directory for inputs and outputs
        self.parser.add_argument('--data_dir', type=str, default='./', help='dataset directory')
        self.parser.add_argument('--out_dir', type=str, default='./results', help='out directory')
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='log directory')

        # model setting
        self.parser.add_argument('--model_name', type=str, help='model name',
                                 choices=['deeperlab', 'deeperlabv3+', 'fcn', 'unet'])

        self.parser.add_argument('--segmentation_only', action='store_true',
                                 help='egmentation only: True, add CAM: False, default is false')
        # model structure
        # which is not supported to change, otherwise may cause some problems
        self.parser.add_argument('--backbone_channels', type=int, default=256)
        self.parser.add_argument('--aspp_channels', type=int, default=256)

        # input size
        self.parser.add_argument('--data_size', type=int, default=512, help='input image size')
        self.parser.add_argument('--batch_size', type=int, default=1)

        # device
        self.parser.add_argument('--device', type=str, default="cuda:0")

        return

    def parse(self):
        if self.opt is not None:
            # check device
            if not torch.cuda.is_available():
                self.opt.device = 'cpu'
                print('Warning: use cpu to run')
        return


class TrainingHyperparameters(BaseHyperparameters):
    """
    training parameters
    """
    def __init__(self):
        super(TrainingHyperparameters, self).__init__()
        self.parse()

    def init_train_params(self):
        # setting intervals
        self.parser.add_argument('--model_intervals', type=int, default=1,
                                 help='interval numbers for saving models')
        self.parser.add_argument('--image_intervals', type=int, default=10,
                                 help='interval numbers for saving images')

        # model prefix
        self.parser.add_argument('--model_prefix', type=int, default=0,
                                 help='prefix epoch of pretraining weights')

        # visdom
        self.parser.add_argument('--use_visdom', action='store_true',
                                 help='use visdom for visualization')

        # training settings
        self.parser.add_argument('--epochs', type=int, default=500, help='total training epochs')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        return

    def parse(self):
        self.initalize()
        self.init_train_params()
        self.opt = self.parser.parse_args()
        super().parse()

        # make dirs
        os.makedirs(os.path.join(self.opt.out_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.opt.out_dir, 'models'), exist_ok=True)
        os.makedirs(self.opt.log_dir, exist_ok=True)

        return


class EvalHyperparameters(BaseHyperparameters):
    """
    evaluation parameters
    """
    def __init__(self):
        super(EvalHyperparameters, self).__init__()
        self.parse()

    def init_eval_params(self):
        # model prefix
        self.parser.add_argument('--model_prefix', type=int, default=0,
                                 help='prefix epoch of pretraining weights')

    def parse(self):
        self.initalize()
        self.init_eval_params()
        self.opt = self.parser.parse_args()
        super().parse()

        # make dirs
        os.makedirs(os.path.join(self.opt.out_dir, 'images'), exist_ok=True)
        os.makedirs(self.opt.log_dir, exist_ok=True)
        return


class InferenceParameters(BaseHyperparameters):
    def __init__(self):
        super(InferenceParameters, self).__init__()
        self.parse()

    def init_inference_params(self):
        self.parser.add_argument('--model_dir', type=str, help='path to model, e.g ./results/50.pt')
        self.parser.add_argument('--model_prefix', type=int,
                                 help='model file prefix, it can be set automatically')

        return

    def parse(self):
        self.initalize()
        self.init_inference_params()
        self.opt = self.parser.parse_args()
        super().parse()

        os.makedirs(self.opt.out_dir, exist_ok=True)
        os.makedirs(self.opt.log_dir, exist_ok=True)

        self.opt.model_prefix = int(os.path.basename(self.opt.model_dir).split('.')[0])

