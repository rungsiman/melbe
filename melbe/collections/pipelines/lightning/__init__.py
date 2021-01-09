import logging
import torch
from abc import ABC
from typing import Dict, Union

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers

from melbe.collections.pipelines.lightning.configs import LightningConfig
from melbe.collections.pipelines.torch import Torch, TorchClasses, \
    MODULE_CLASS, MODELS_CLASS, OPTIMIZERS_CLASS, SCHEDULERS_CLASS
from melbe.configs import ClassConfig, MelbeConfig, select
from melbe.monitoring import TimeMonitor


LIGHTNING_CLASS = Union[LightningModule, None]


class LightningClasses(TorchClasses):
    lighting: LIGHTNING_CLASS = ...

    def __init__(self, lightning: LIGHTNING_CLASS = None, **kwargs):
        super().__init__(**kwargs)
        self.lighting = lightning


class Lightning(Torch, ABC):
    config: LightningConfig = ...

    lightning: LightningModule = ...
    trainer: pl.Trainer = ...

    def __init__(self,
                 melbe_config: MelbeConfig,
                 lightning: LIGHTNING_CLASS = None,
                 models: MODELS_CLASS = None,
                 optimizers: OPTIMIZERS_CLASS = None,
                 schedulers: SCHEDULERS_CLASS = None,
                 **kwargs):
        super().__init__(melbe_config, **kwargs)
        self.config = LightningConfig(melbe_config=melbe_config, **select(kwargs, LightningConfig.name))
        self.classes = LightningClasses(lightning, models=models, optimizers=optimizers, schedulers=schedulers)

    def setup(self):
        self.setup_module('models', required=True)
        self.setup_module('optimizers', required=True)
        self.setup_module('schedulers')

        tb_logger = pl_loggers.TensorBoardLogger(self.config.paths.store.logs)
        self.trainer = pl.Trainer(logger=tb_logger,
                                  **{'max_epochs': self.config.epochs,
                                     'log_gpu_memory': True,
                                     **self.config.trainer})

        # json.dump(self.config, open(os.path.join(tb_logger.log_dir, 'config.json'), 'w'))
        return self

    def setup_module(self, name: str, required: bool = False) -> None:
        class_module: MODULE_CLASS = getattr(self.classes, name)
        config_module: Union[ClassConfig, Dict[ClassConfig]] = getattr(self.config, name)

        if class_module is not None:
            if isinstance(class_module, dict):
                for key, comp in class_module.items():
                    config_module[key].override(comp)
            else:
                config_module.override(class_module)

        if required and not isinstance(config_module.cls, dict) and config_module.cls is None:
            error = f'At least one module of "{name}" is required but none provided.'
            logging.error(error)
            raise RuntimeError(error)

    def prepare(self):
        self.build_inputs()
        self.build_data_loaders()
        return self

    def train(self):
        timer = TimeMonitor()

        if self.data_loaders.validate is None:
            logging.info('Training without validation...')
            self.trainer.fit(self.lightning, self.data_loaders.train)
        else:
            logging.info('Training with validation...')
            self.trainer.fit(self.lightning, self.data_loaders.train, self.data_loaders.validate)

        logging.info(f'Done training in {timer.step().string}.')
        return self

    def test(self):
        timer = TimeMonitor()
        logging.info('Testing...')
        result = self.trainer.test(test_dataloaders=self.data_loaders.test)
        logging.info(f'Done testing in {timer.step().string}:\n{result}')
        return self

    def predict(self, **kwargs):
        super().predict(**kwargs)
        timer = TimeMonitor()
        logging.info('Predicting...')

        logits = self.lightning(input_ids=self.inputs.predict.input_ids,
                                attention_mask=self.inputs.predict.attention_mask).logits.detach().cpu()
        self.predictions.logits = logits

        preds = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1).detach().cpu()
        self.process_predictions(preds)

        logging.info(f'Done predicting in {timer.step().string}.')
        return self.predictions


class PipelineLightning(LightningModule):
    def __init__(self,
                 models: Union[ClassConfig, Dict[str, ClassConfig]],
                 optimizers: Union[ClassConfig, Dict[str, ClassConfig]],
                 schedulers: Union[ClassConfig, Dict[str, ClassConfig]] = None):
        super().__init__()

        if isinstance(models, dict):
            self.models = {}
            for key, model in models.items():
                self.models[key] = self.init_model(model)
        else:
            self.models = self.init_model(models)

        self.pipeline_optimizers = optimizers
        self.pipeline_schedulers = schedulers
        self.optimizers_order = []

    def init_model(self, model: ClassConfig):
        return model.cls(**model.kwargs)

    def configure_optimizers(self):
        if isinstance(self.pipeline_optimizers, dict):
            optimizers, schedulers = [], []

            for key in self.pipeline_optimizers.keys():
                optimizers.append(self.pipeline_optimizers[key](self.models[key].parameters(),
                                                                **self.pipeline_optimizers[key].kwargs))

                if key in self.pipeline_schedulers:
                    schedulers.append(self.pipeline_schedulers[key](optimizers[-1],
                                                                    **self.pipeline_schedulers[key].kwargs))
                self.optimizers_order.append(key)

            return optimizers, schedulers

        else:
            optimizer = self.pipeline_optimizers(self.models.parameters(), **self.pipeline_optimizers.kwargs)

            if self.pipeline_schedulers.cls is not None:
                scheduler = self.pipeline_schedulers(optimizer, **self.pipeline_schedulers.kwargs)
                return [optimizer], [scheduler]

            return optimizer

    def get_progress_bar_dict(self):
        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
