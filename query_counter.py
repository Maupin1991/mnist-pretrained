from typing import TypeVar
from foolbox import PyTorchModel

T = TypeVar("T")

class PytorchModelCounter(PyTorchModel):
    def __init__(self, *args, **kwargs):
        self._query_count = 0
        super(PytorchModelCounter, self).__init__(*args, **kwargs)

    def __call__(self, inputs: T) -> T:
        self._query_count += inputs.shape[0]
        return super(PytorchModelCounter, self).__call__(inputs)

    def reset_counter(self):
        self._query_count = 0

    @property
    def query_count(self):
        return self._query_count
