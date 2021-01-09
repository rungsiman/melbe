from typing import Any, Dict, Union

from melbe.configs import ClassConfig, Config
from melbe.configs import init, select
from melbe.pipelines import PipelineConfig


class TorchDataLoadersConfig(Config):
    train: Dict[str, Any] = ...
    validate: Dict[str, Any] = ...
    test: Dict[str, Any] = ...
    predict: Dict[str, Any] = ...

    def __init__(self, **kwargs):
        super().__init__(**init(kwargs))
        self.train = select(kwargs, 'train')
        self.validate = select(kwargs, 'validate')
        self.test = select(kwargs, 'test')
        self.predict = select(kwargs, 'predict')


class TorchConfig(PipelineConfig):
    name = 'torch'
    version = '0.1.0'

    models: Union[ClassConfig, Dict[str, ClassConfig], None] = ...
    optimizers: Union[ClassConfig, Dict[str, ClassConfig], None] = ...
    schedulers: Union[ClassConfig, Dict[str, ClassConfig], None] = ...
    data_loaders: TorchDataLoadersConfig = ...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_loaders = TorchDataLoadersConfig(**select(kwargs, 'data_loaders'))

        if 'models' in kwargs:
            self.models = {}
            for key in kwargs['models'].split(','):
                self.models[key] = ClassConfig(**select(kwargs, f'model-{key}'))
        else:
            self.models = ClassConfig(**(select(kwargs, 'model')))

        if 'optimizers' in kwargs:
            self.optimizers = {}
            for key in kwargs['optimizers'].split(','):
                self.optimizers[key] = ClassConfig(**select(kwargs, f'optimizer-{key}'))
        else:
            self.optimizers = ClassConfig(**(select(kwargs, 'optimizer')))

        if 'schedulers' in kwargs:
            self.schedulers = {}
            for key in kwargs['schedulers'].split(','):
                self.schedulers[key] = ClassConfig(**select(kwargs, f'scheduler-{key}'))
        else:
            self.schedulers = ClassConfig(**(select(kwargs, 'scheduler')))
