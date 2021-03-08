from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.fcn import FCN4x
from models.deeperlab import DeeperLabC
from models.classifier import Classifier


def model_choice(model_name, out_channels=2):
    model_name = model_name.lower()

    model_dict = {'unet': UNet(out_channels),
                  'deeplabv3+': DeepLabV3Plus(out_channels),
                  'deeplabv3plus': DeepLabV3Plus(out_channels),
                  'fcn': FCN4x(out_channels),
                  'deeperlabc': DeeperLabC(out_channels),
                  'deeperlab': DeeperLabC(out_channels)
                  }

    try:
        model = model_dict[model_name]
    except KeyError:
        model = None
        print('no such model, please check "model_name" in config.py')
        exit(0)

    classifier = Classifier(out_channels)
    return model, classifier


