"""
Model trained with https://github.com/louis2889184/pytorch-adversarial-training
"""
import foolbox as fb
import torch

from model_madry import Model


def create():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_path = fb.zoo.fetch_weights(
        'https://github.com/maurapintor/mnist-pretrained/releases/download/v1.2/madry_mnist.pt',
        unzip=False
    )
    model = Model(i_c=1, n_c=10)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessing = None

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1),
                                    preprocessing=preprocessing,
                                    device=device)

    return fmodel
