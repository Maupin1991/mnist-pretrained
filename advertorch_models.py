import foolbox as fb
import torch
from advertorch.test_utils import LeNet5

model_links = {
    'adv_trained': 'https://github.com/BorealisAI/advertorch/raw/master/advertorch_examples/trained_models'
                   '/mnist_lenet5_advtrained.pt',
    'cln_trained': 'https://github.com/BorealisAI/advertorch/raw/master/advertorch_examples/trained_models'
                   '/mnist_lenet5_clntrained.pt'
}


def create(adv_train=True):
    if adv_train:
        model_link = model_links['adv_trained']
    else:
        model_link = model_links['cln_trained']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_path = fb.zoo.fetch_weights(
        model_link,
        unzip=False
    )
    model = LeNet5()
    model.to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preprocessing = {'mean': 0.5,
                     'std': 0.5}

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1),
                                    preprocessing=preprocessing,
                                    device=device)

    return fmodel
