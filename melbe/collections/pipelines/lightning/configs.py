from typing import Any, Dict

from melbe.collections.pipelines.torch.configs import TorchConfig
from melbe.configs import select


class LightningConfig(TorchConfig):
    name = 'lightning'
    version = '0.1.0'

    lightning: Dict[str, Any] = ...
    trainer: Dict[str, Any] = ...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lightning = select(kwargs, 'lightning')
        self.trainer = select(kwargs, 'trainer')
