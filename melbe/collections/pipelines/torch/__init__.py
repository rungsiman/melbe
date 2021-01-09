import logging
from abc import ABC
from typing import Any, Dict, Type, Union

from torch.optim import Optimizer
from torch.tensor import Tensor
from torch.utils.data import DataLoader, TensorDataset

from melbe.collections.pipelines.torch.configs import TorchConfig
from melbe.data import PREDICTIONS, TEXT_SENTENCE, LIST_SENTENCE
from melbe.configs import ClassConfig, MelbeConfig, select
from melbe.pipelines import Pipeline, PipelineInputs, PipelinePredictions


MODELS_CLASS = SCHEDULERS_CLASS = MODULE_CLASS = Union[Type, Dict[str, Type], None]
OPTIMIZERS_CLASS = Union[Type[Optimizer], Dict[str, Type[Optimizer]], None]


class TorchInputs:
    input_ids: Tensor = ...
    attention_mask: Tensor = ...
    labels: Tensor = ...

    def __init__(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


class TorchPredictions(PipelinePredictions):
    logits: Tensor = ...
    preds: Tensor = ...

    def __init__(self, logits: Tensor = None, preds: Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits = logits
        self.preds = preds


class TorchPipelineInputs(PipelineInputs):
    train: TorchInputs = ...
    validate: TorchInputs = ...
    test: TorchInputs = ...
    predict: TorchInputs = ...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TorchDataLoaders:
    train: DataLoader = ...
    validate: DataLoader = ...
    test: DataLoader = ...
    predict: DataLoader = ...

    def __init__(self,
                 train: DataLoader = None,
                 validate: DataLoader = None,
                 test: DataLoader = None,
                 predict: DataLoader = None):
        self.train = train
        self.validate = validate
        self.test = test
        self.predict = predict


class TorchClasses:
    models: MODELS_CLASS = ...
    optimizers: OPTIMIZERS_CLASS = ...
    schedulers: SCHEDULERS_CLASS = ...

    def __init__(self,
                 models: MODELS_CLASS = None,
                 optimizers: OPTIMIZERS_CLASS = None,
                 schedulers: SCHEDULERS_CLASS = None):
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers


class Torch(Pipeline, ABC):
    config: TorchConfig = ...
    classes: TorchClasses = ...

    models: Union[Any, Dict[str, Any]] = ...
    optimizers: Union[Optimizer, Dict[str, Optimizer]] = ...
    schedulers: Union[Any, Dict[str, Any], None] = ...

    inputs: TorchPipelineInputs = None
    data_loaders: TorchDataLoaders = None
    predictions: TorchPredictions = None

    checkpoint: Dict[str, Any] = ...

    def __init__(self,
                 melbe_config: MelbeConfig,
                 models: MODELS_CLASS = None,
                 optimizers: OPTIMIZERS_CLASS = None,
                 schedulers: SCHEDULERS_CLASS = None,
                 **kwargs):
        super().__init__(melbe_config)
        self.config = TorchConfig(melbe_config=melbe_config, **select(kwargs, TorchConfig.name))
        self.classes = TorchClasses(models, optimizers, schedulers)

    def setup(self):
        self.setup_module('models', required=True)
        self.setup_module('optimizers', required=True)
        self.setup_module('schedulers')
        return self

    def setup_module(self, name: str, required: bool = False) -> None:
        factory_module: MODULE_CLASS = getattr(self.classes, name)
        config_module: Union[ClassConfig, Dict[ClassConfig]] = getattr(self.config, name)

        if factory_module is not None:
            if isinstance(factory_module, dict):
                for key, comp in factory_module.items():
                    config_module[key].override(comp)
            else:
                config_module.override(factory_module)

        if isinstance(config_module, dict):
            setattr(self, name, {})
            for key, comp in config_module.items():
                getattr(self, name)[key] = comp(**comp.kwargs)

        elif config_module.cls is not None:
            setattr(self, name, config_module(**config_module.kwargs))

        elif required:
            error = f'At least one module of "{name}" is required but none provided.'
            logging.error(error)
            raise RuntimeError(error)

        else:
            setattr(self, name, None)

    def prepare(self):
        self.build_inputs()
        self.build_data_loaders()
        return self

    def build_data_loaders(self):
        def build(inputs: TorchInputs) -> TensorDataset:
            # input_ids: torch.LongTensor of shape (batch_size, sequence_length)
            # attention_mask: torch.FloatTensor of shape (batch_size, sequence_length)
            # labels: torch.LongTensor of shape (batch_size, sequence_length)
            if inputs.labels is not None:
                return TensorDataset(inputs.input_ids,
                                     inputs.attention_mask,
                                     inputs.labels)
            else:
                return TensorDataset(inputs.input_ids,
                                     inputs.attention_mask)

        logging.info('Building data loaders...')
        self.data_loaders = TorchDataLoaders()

        if self.inputs.predict is not None:
            self.data_loaders.predict = DataLoader(build(self.inputs.predict), **self.config.data_loaders.predict)
        else:
            if 'train' in self.tasks and self.inputs.train is not None:
                self.data_loaders.train = DataLoader(build(self.inputs.train), **self.config.data_loaders.train)

                if self.inputs.validate is not None:
                    self.data_loaders.validate = DataLoader(build(self.inputs.validate),
                                                            **self.config.data_loaders.validate)

            else:
                self.data_loaders.train = None
                self.data_loaders.validate = None

            if 'test' in self.tasks and self.inputs.test is not None:
                self.data_loaders.test = DataLoader(build(self.inputs.test), **self.config.data_loaders.test)

        logging.info('Building data loaders completed.')
        return self

    def predict(self, **kwargs) -> TorchPredictions:
        super().predict(**kwargs)
        return self.predictions

    def process_predictions(self, preds: Tensor) -> None:
        if self.predictions.type == 'text' or self.predictions.type == 'words':
            self.predictions.results = self.cast_prediction_sentence(self.documents[0][0], preds[0], 0)

        elif self.predictions.type == 'document':
            self.predictions.results = [self.cast_prediction_sentence(self.documents[0][i], preds[i], i)
                                        for i in range(len(preds))]

        elif self.predictions.type == 'documents':
            self.predictions.results, pred_i = [], 0

            for document in self.documents:
                self.predictions.results.append([])

                for sentence in document:
                    self.predictions.results[-1].append(self.cast_prediction_sentence(sentence,
                                                                                      preds[pred_i],
                                                                                      pred_i))
                    pred_i += 1

    def cast_prediction_sentence(self,
                                 sentence: Union[TEXT_SENTENCE, LIST_SENTENCE],
                                 preds: Tensor,
                                 pred_i: int) -> PREDICTIONS:
        ...
