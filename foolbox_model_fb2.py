import foolbox as fb
import torch
from foolbox.models import PyTorchModel

from model import MNISTModel


def create():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_path = fb.zoo.fetch_weights(
        'https://github.com/Maupin1991/mnist-pretrained/releases/download/v1.1/mnist_cnn.pt',
        unzip=False
    )
    model = MNISTModel()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessing = {'mean': 0.5,
                     'std': 0.5}

    fmodel = PyTorchModel(model, bounds=(0, 1),
                          num_classes=10,
                          preprocessing=preprocessing, channel_axis=1)

    return fmodel
