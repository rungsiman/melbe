import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union

from melbe.components import init_component
from melbe.configs import Config, MelbeConfig


TEXT_SENTENCE = Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]
LIST_SENTENCE = Dict[str, Union[List[str], List[Dict[str, Union[str, int]]]]]
TEXT_DOCUMENT = List[TEXT_SENTENCE]
TEXT_DOCUMENTS = List[TEXT_DOCUMENT]
LIST_DOCUMENT = List[LIST_SENTENCE]
LIST_DOCUMENTS = List[LIST_DOCUMENT]
DOCUMENT = Union[TEXT_DOCUMENT, LIST_DOCUMENT, None]
DOCUMENTS = Union[TEXT_DOCUMENTS, LIST_DOCUMENTS, None]
PREDICTIONS = Union[Dict[str, Union[str, int]],
                    List[Dict[str, Union[str, int]]],
                    List[List[Dict[str, Union[str, int]]]]]


class Labels:
    tags: Dict[str, id] = None
    ids: Dict[id, str] = None

    def __init__(self):
        self.tags = {}
        self.ids = {}

    def tag(self, key: str) -> int:
        if key not in self.tags:
            self.tags[key] = tid = len(self.tags)
            self.ids[tid] = key
        return self.tags[key]

    def load(self, obj: Dict[str, Union[int, str]]):
        self.tags = obj

        for tag, tid in obj.items():
            self.ids[tid] = tag

        return self


class DatasetConfig(Config):
    version: str = ...
    no_class_tag: str = '--NCT--'


class Dataset(ABC):
    _melbe_meta: str = 'dataset'
    config: DatasetConfig = ...

    documents_train: DOCUMENTS = None
    documents_test: DOCUMENTS = None
    labels: Labels = ...

    init_clean_static: bool = ...
    init_skip_preparation: bool = ...

    def __init__(self,
                 melbe_config: MelbeConfig,
                 clean_non_static: bool = False,
                 skip_preparation: bool = False, **kwargs):
        self.melbe_config = melbe_config
        self.labels = Labels()
        self.init_clean_static = clean_non_static
        self.init_skip_preparation = skip_preparation

    @abstractmethod
    def check(self) -> bool:
        """Check whether the locally required dataset exists"""
        ...

    @abstractmethod
    def preprocess(self):
        """Change the dataset into a standardized format"""
        ...

    @abstractmethod
    def postprocess(self, predictions: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        ...

    @abstractmethod
    def save(self):
        """Save the processed dataset"""
        ...

    @abstractmethod
    def load(self) -> bool:
        """Load the processed dataset"""
        ...

    @abstractmethod
    def clean(self, full_clean: bool = False, **kwargs):
        ...

    @abstractmethod
    def retrieve(self) -> bool:
        """Retrieve the dataset from an external source"""
        ...

    def post_init(self, **kwargs):
        if self.init_clean_static:
            self.clean(**kwargs)

        if not self.init_skip_preparation:
            self.prepare()

    def prepare(self):
        if self.load():
            return self

        elif self.check() or self.retrieve():
            self.preprocess()
            self.build_labels()
            self.save()
            return self

        else:
            error = f'Could not find or retrieve dataset "{self.config.name}." ' \
                    f"Check the dataset's class's path configuration for the required files and locations."
            logging.error(error)
            raise FileNotFoundError(error)

    def build_labels(self):
        self.labels.tag(self.config.no_class_tag)

        if self.documents_train is not None:
            self.build_document_labels(self.documents_train, self.labels)
        if self.documents_test is not None:
            self.build_document_labels(self.documents_test, self.labels)

        return self

    @staticmethod
    def build_document_labels(documents: DOCUMENTS, labels: Labels):
        for document in documents:
            for sentence in document:
                if 'mentions' in sentence:
                    for mention in sentence['mentions']:
                        labels.tag(mention['tag'])

    def chop(self, retain_train: float = 1.0, retain_test: float = 1.0, skip_build_labels: bool = False):
        self.documents_train = self.documents_train[:int(len(self.documents_train) * retain_train)]
        self.documents_test = self.documents_test[:int(len(self.documents_test) * retain_test)]

        if not skip_build_labels:
            self.labels = Labels()
            self.build_labels()

        return self

    def shuffle(self, train: bool = True, test: bool = True):
        if train:
            self.documents_train = self.shuffle_documents(self.documents_train)

        if test:
            self.documents_test = self.shuffle_documents(self.documents_test)

        return self

    @staticmethod
    def shuffle_documents(documents: DOCUMENTS) -> DOCUMENTS:
        order = [i for i in range(len(documents))]
        random.shuffle(order)
        return [documents[i] for i in order]

    def fetch(self, task: str) -> DOCUMENTS:
        return self.documents_train if task == 'train' else self.documents_test


def init_dataset(melbe_config: MelbeConfig,
                 global_kwargs: Dict[str, Any],
                 dataset: Type[Dataset] = None,
                 **kwargs):
    return init_component(melbe_config,
                          global_kwargs,
                          parents=[melbe_config.paths.components.datasets, 'melbe.collections.datasets'],
                          branch=global_kwargs['dataset'] if 'dataset' in global_kwargs else None,
                          custom_cls=dataset,
                          **kwargs)
