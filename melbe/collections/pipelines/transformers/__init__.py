import logging
from typing import Dict, Type, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.tensor import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from melbe.collections.pipelines.torch import TorchInputs, TorchPipelineInputs, OPTIMIZERS_CLASS, SCHEDULERS_CLASS
from melbe.collections.pipelines.lightning import Lightning, LightningClasses, PipelineLightning, LIGHTNING_CLASS
from melbe.collections.pipelines.transformers.configs import TransformersConfig, ModelConfig
from melbe.configs import ClassConfig, MelbeConfig, select


MODELS_CLASS = Union[Type[PreTrainedModel], Dict[str, Type[PreTrainedModel]], None]
TOKENIZER_CLASS = Union[Type[PreTrainedTokenizer], None]


class TransformersInputs(TorchInputs):
    offsets: Tensor = ...

    def __init__(self, offsets: Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.offsets = offsets


class TransformersPipelineInputs(TorchPipelineInputs):
    train: TransformersInputs = ...
    validate: TransformersInputs = ...
    test: TransformersInputs = ...
    predict: TransformersInputs = ...


class TransformersClasses(LightningClasses):
    tokenizer: TOKENIZER_CLASS = ...

    def __init__(self, tokenizer: TOKENIZER_CLASS = None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer


class Transformers(Lightning):
    config: TransformersConfig = ...
    classes: TransformersClasses = ...
    tokenizer: Union[PreTrainedTokenizer, None] = ...

    inputs: TransformersPipelineInputs = None

    def __init__(self,
                 melbe_config: MelbeConfig,
                 lightning: LIGHTNING_CLASS = None,
                 models: MODELS_CLASS = None,
                 optimizers: OPTIMIZERS_CLASS = None,
                 schedulers: SCHEDULERS_CLASS = None,
                 tokenizer: TOKENIZER_CLASS = None,
                 **kwargs):
        super().__init__(melbe_config, **kwargs)
        self.config = TransformersConfig(melbe_config=melbe_config, **select(kwargs, TransformersConfig.name))
        self.classes = TransformersClasses(tokenizer, lightning=lightning, models=models,
                                           optimizers=optimizers, schedulers=schedulers)

    def setup(self):
        super().setup()

        if self.classes.tokenizer is not None:
            self.tokenizer = self.classes.tokenizer \
                .from_pretrained(self.config.tokenizer.pretrained, **self.config.tokenizer.kwargs)
        elif self.config.tokenizer.cls is not None:
            self.tokenizer = self.config.tokenizer.cls \
                .from_pretrained(self.config.tokenizer.pretrained, **self.config.tokenizer.kwargs)

        return self

    def prepare(self):
        super().prepare()
        self.lightning = TransformersLightning(models=self.config.models,
                                               optimizers=self.config.optimizers,
                                               schedulers=self.config.schedulers,
                                               num_labels=len(self.labels.tags),
                                               **self.config.lightning)
        return self

    def transform(self, documents):
        if documents is None or len(documents) == 0:
            return None

        logging.info(f'Tokenizing {len(documents)} documents...')
        input_sentences, input_labels = [], []
        sentence_type = 'words' if 'words' in documents[0][0] else 'text'

        for document in documents:
            for sentence in document:
                input_sentences.append(sentence[sentence_type])

        # Tokenized inputs contain the following attributes:
        #   - input_ids: Tensor         (padded right)
        #   - token_type_ids: Tensor    (all zero since we are not doing question-answering)
        #   - attention_mask: Tensor    (1 for non-padded tokens, 0 otherwise)
        #   - length: Tensor            (a tensor of size all(sentences) containing the same length)
        #   - offset_mapping            ("text" type only)
        inputs: BatchEncoding = self.tokenizer(input_sentences,
                                               padding='max_length',
                                               truncation=True,
                                               return_length=True,
                                               return_tensors='pt',
                                               is_split_into_words=sentence_type == 'words',
                                               return_offsets_mapping=sentence_type == 'text',
                                               **self.config.tokenizer.kwargs)

        logging.info('Done.\n'
                     f'.. Input IDs shape: {list(inputs.input_ids.shape)}\n'
                     f'.. Attention masks shape: {list(inputs.attention_mask.shape)}')

        results = TransformersInputs(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

        if 'mentions' in documents[0][0]:
            logging.info(f'Tagging labels (dictionary size: {len(self.labels.tags)})...')
            max_length = inputs.length[0]

            if sentence_type == 'words':
                for document in documents:
                    for sentence in document:
                        input_labels.append([self.labels.tag(self.no_class_tag)] * max_length)

                        for mention in sentence['mentions']:
                            for i in range(mention['start'], mention['end'] + 1):
                                input_labels[-1][i] = self.labels.tag(mention['tag'])

            else:
                ...     # inputs.offset_mapping

            labels_tensor = torch.tensor(input_labels)
            logging.info('Tagging labels completed.\n'
                         f'.. Label tensor shape: {list(labels_tensor.shape)}')

            results.labels = labels_tensor

        if sentence_type == 'text':
            results.offsets = inputs.offset_mapping

        logging.info('Tokenization completed.')
        return results

    def train(self):
        super().train()
        return self

    def test(self):
        super().test()
        return self

    def predict(self, **kwargs):
        return super().predict(**kwargs)

    def cast_prediction_sentence(self, sentence, preds, pred_i):
        boundary = self.inputs.predict.attention_mask[pred_i].sum() - 1
        matches = []

        if 'text' in sentence:
            offsets = self.inputs.predict.offsets[pred_i][1:boundary]
            i = 1

            while i < boundary:
                if preds[i] != self.labels.tags[self.no_class_tag]:
                    j = i

                    while j < boundary and preds[i] == preds[j]:
                        j += 1

                    offset = (int(offsets[i][0]), int(offsets[j - 1][1]))
                    matches.append({'offset': offset,
                                    'text': sentence['text'][offset[0]:offset[1]],
                                    'label': self.labels.ids[int(preds[i])]})

                    i = j
                i += 1

        else:
            ...

        return matches

    def save(self):
        super().save()
        return self

    def load(self) -> bool:
        return super().load()


class TransformersLightning(PipelineLightning):
    def __init__(self,
                 num_labels: int,
                 models: ModelConfig,
                 optimizers: ClassConfig,
                 schedulers: ClassConfig = None,
                 **kwargs):
        self.num_labels = num_labels
        super().__init__(models, optimizers, schedulers)

        # Add one to num_labels to account for the default class (words with no labels)
        self.train_f1 = pl.metrics.F1(num_classes=num_labels + 1)
        self.valid_f1 = pl.metrics.F1(num_classes=num_labels + 1)
        self.test_f1 = pl.metrics.F1(num_classes=num_labels + 1)
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def init_model(self, model: ModelConfig):
        config = model.config.cls.from_pretrained(model.pretrained, **model.config.kwargs)
        config.update({**model.config.override, 'num_labels': self.num_labels + 1})
        return model.cls.from_pretrained(model.pretrained, config=config, **model.kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        input_ids, attention_mask, labels = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        preds = torch.argmax(F.log_softmax(output.logits, dim=-1), dim=-1)
        f1 = self.train_f1(preds, labels)
        acc = self.train_acc(preds, labels)
        self.log('train_loss', output.loss)
        self.log('train_f1_step', f1)
        self.log('train_acc_step', acc)

        return output.loss

    def training_epoch_end(self, outs):
        self.log('train_f1_epoch', self.train_f1.compute())
        self.log('train_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        preds = torch.argmax(F.log_softmax(output.logits, dim=-1), dim=-1)
        f1 = self.valid_f1(preds, labels)
        acc = self.valid_acc(preds, labels)
        self.log('valid_loss', output.loss)
        self.log('valid_f1_step', f1)
        self.log('valid_acc_step', acc)

    def validation_epoch_end(self, outputs):
        self.log('valid_f1_epoch', self.valid_f1.compute())
        self.log('valid_acc_epoch', self.valid_acc.compute())

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        preds = torch.argmax(F.log_softmax(output.logits, dim=-1), dim=-1)
        self.test_f1(preds, labels)
        self.test_acc(preds, labels)

    def test_epoch_end(self, outputs):
        self.log('test_f1', self.test_f1.compute())
        self.log('test_acc', self.test_acc.compute())

    def forward(self, input_ids, attention_mask, labels=None):
        return self.models(input_ids=input_ids.to(self.device),
                           attention_mask=attention_mask.to(self.device),
                           labels=labels.to(self.device) if labels is not None else None)
