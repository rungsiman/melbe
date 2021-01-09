from typing import Any, Dict, Union

from melbe.collections.pipelines.lightning.configs import LightningConfig
from melbe.configs import ClassConfig
from melbe.configs import init, select


class PretrainedConfig(ClassConfig):
    override: Dict[str, Any] = ...

    def __init__(self, **kwargs):
        super().__init__(**init(kwargs))
        self.override = {**select(kwargs, 'override')}


class ModelConfig(ClassConfig):
    pretrained: str = ...
    config: PretrainedConfig = ...

    def __init__(self, **kwargs):
        super().__init__(**init(kwargs))
        self.config = PretrainedConfig(**select(kwargs, 'config'))


class TokenizerConfig(ClassConfig):
    pretrained: str = ...


class TransformersConfig(LightningConfig):
    name = 'transformers'
    version = '0.1.0'

    models: Union[ModelConfig, Dict[str, ModelConfig], None] = None
    tokenizer: Union[TokenizerConfig, None] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = TokenizerConfig(**select(kwargs, 'tokenizer'))
