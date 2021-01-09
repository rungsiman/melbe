import os

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

from melbe.components import init_component
from melbe.configs import Config, MelbeConfig, MelbePathConfig
from melbe.configs import init, mkdir, select
from melbe.data import Dataset, TEXT_SENTENCE, LIST_SENTENCE, DOCUMENT, DOCUMENTS, PREDICTIONS, Labels


class PipelinePathConfig(Config):
    name = 'paths'

    class Store(Config):
        name = 'store'

        def __init__(self, config: MelbePathConfig, pipeline: str, **kwargs):
            super().__init__(**init(kwargs))
            self.root = mkdir(os.path.join(config.data.store.pipelines, pipeline))
            self.logs = mkdir(os.path.join(self.root, 'logs'))

    def __init__(self, config: MelbePathConfig, pipeline: str, **kwargs):
        super().__init__(**init(kwargs))
        self.store = PipelinePathConfig.Store(config, pipeline, **select(kwargs, PipelinePathConfig.Store.name))


class PipelineConfig(Config):
    version: str = ...
    paths: PipelinePathConfig = ...
    epochs: int = ...
    validation_ratio: float = ...

    def __init__(self, melbe_config: MelbeConfig, **kwargs):
        super().__init__(**init(kwargs))
        self.paths = PipelinePathConfig(melbe_config.paths, self.name, **select(kwargs, PipelinePathConfig.name))


class PipelineInputs:
    train: Union[Any, None] = ...
    validate: Union[Any, None] = ...
    test: Union[Any, None] = ...
    predict: Union[Any, None] = ...

    def __init__(self, train: Any = None, validate: Any = None, test: Any = None, predict: Any = None):
        self.train = train
        self.validate = validate
        self.test = test
        self.predict = predict


class PipelinePredictions:
    type: str = ...
    results: PREDICTIONS = ...

    def __init__(self, prediction_type: str = None, results: PREDICTIONS = None):
        self.prediction_type = prediction_type
        self.results = results


class Pipeline(ABC):
    _melbe_meta: str = 'pipeline'

    config: PipelineConfig = ...
    melbe_config: MelbeConfig = ...

    dataset: Union[Dataset, None] = None
    documents: Union[DOCUMENTS, Dict[str, DOCUMENTS], None] = None
    labels: Labels = ...
    no_class_tag: str = '--NCT--'

    tasks: List[str] = ...
    inputs: PipelineInputs = ...
    predictions: PipelinePredictions = ...

    def __init__(self,
                 melbe_config: MelbeConfig,
                 tasks: Union[str, List[str]] = ('train', 'test'),
                 no_class_tag: str = '--NCT--',
                 *args, **kwargs):
        self.melbe_config = melbe_config
        self.tasks = tasks
        self.no_class_tag = no_class_tag
        self.predictions = PipelinePredictions()

    @abstractmethod
    def setup(self):
        ...

    def prepare(self):
        self.build_inputs()
        return self

    @abstractmethod
    def transform(self, documents: DOCUMENTS) -> Any:
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self) -> bool:
        ...

    def fetch(self,
              data: Union[Dataset, TEXT_SENTENCE, LIST_SENTENCE, DOCUMENT, DOCUMENTS],
              labels: Union[Labels, None] = None,
              no_class_tag: Union[str, None] = None):
        if isinstance(data, Dataset):
            self.dataset = data
            self.labels = data.labels
            self.no_class_tag = data.config.no_class_tag
        else:
            self.documents = data
            self.labels = labels

            if no_class_tag is not None:
                self.no_class_tag = no_class_tag

        self.setup().prepare()
        return self

    def build_inputs(self):
        def build(documents_train: DOCUMENTS, documents_test: DOCUMENTS) -> None:
            if 'train' in self.tasks:
                divider = int((1 - self.config.validation_ratio) * len(documents_train))

                if documents_train is not None and len(documents_train):
                    self.inputs.train = self.transform(documents_train[:divider])
                else:
                    self.inputs.train = None

                if divider < len(documents_train):
                    self.inputs.validate = self.transform(documents_train[divider:])
                else:
                    self.inputs.validate = None

            if 'test' in self.tasks:
                if documents_test is not None and len(documents_test):
                    self.inputs.test = self.transform(documents_test)
                else:
                    self.inputs.test = None

        self.inputs = PipelineInputs()

        if isinstance(self.dataset, Dataset):
            build(self.dataset.fetch('train'), self.dataset.fetch('test'))
        else:
            build(self.documents['train'] if isinstance(self.documents, dict) else self.documents,
                  self.documents['test'] if isinstance(self.documents, dict) else self.documents)
        return self

    def predict(self,
                sentence: Union[str, List[str]] = None,
                document: DOCUMENT = None,
                documents: DOCUMENTS = None) -> PipelinePredictions:
        self.inputs = PipelineInputs()
        self.predictions = PipelinePredictions()

        if sentence is not None:
            if isinstance(sentence, str):
                self.documents = [[{'text': sentence}]]
                self.inputs.predict = self.transform(self.documents)
                self.predictions.type = 'text'
            else:
                self.documents = [[{'words': sentence}]]
                self.inputs.predict = self.transform(self.documents)
                self.predictions.type = 'words'

        elif document is not None:
            self.documents = [document]
            self.inputs.predict = self.transform(self.documents)
            self.predictions.type = 'document'

        else:
            self.documents = documents
            self.inputs.predict = self.transform(self.documents)
            self.predictions.type = 'documents'

        return ...


def init_pipeline(melbe_config: MelbeConfig,
                  global_kwargs: Dict[str, Any],
                  pipeline: Type[Pipeline] = None,
                  **kwargs):
    return init_component(melbe_config,
                          global_kwargs,
                          parents=[melbe_config.paths.components.pipelines, 'melbe.collections.pipelines'],
                          branch=global_kwargs['pipeline'] if 'pipeline' in global_kwargs else None,
                          custom_cls=pipeline,
                          **kwargs)
