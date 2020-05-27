import foolbox as fb
import torch

from model import MNISTModel


def create():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_path = fb.zoo.fetch_weights(
        '',
        unzip=False
    )

    model = MNISTModel()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessing = {'mean': 0.1307,
                     'std': 0.3081}

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1),
                                    preprocessing=preprocessing,
                                    device=device)

    return fmodel
