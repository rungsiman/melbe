import json
import logging
import os
import shutil
import urllib.request
import zipfile
from typing import Dict

from melbe.collections.datasets.aida.configs import AidaConfig
from melbe.configs import MelbeConfig, check_paths, init, select
from melbe.data import Dataset, DOCUMENTS
from melbe.monitoring import TimeMonitor


class Aida(Dataset):
    config: AidaConfig = ...

    documents_test_a: DOCUMENTS = None
    documents_test_b: DOCUMENTS = None

    def __init__(self, melbe_config: MelbeConfig, **kwargs):
        super().__init__(melbe_config, **init(kwargs))
        self.config: AidaConfig = AidaConfig(melbe_config, **select(kwargs, AidaConfig.name))
        self.entity_map: Dict[str, Dict[str, str]] = {}
        self.post_init(**kwargs)

    def check(self):
        return os.path.exists(self.config.paths.data.static.all)

    def preprocess(self):
        logging.info('Processing AIDA-YAGO2 dataset...')
        timer = TimeMonitor().begin()
        test_a_i, test_b_i = None, None

        with open(self.config.paths.data.static.all) as reader:
            lines = reader.readlines()
            full, mention = [], None

            for i in range(len(lines)):
                if '-DOCSTART-' in lines[i]:
                    if '947testa CRICKET' in lines[i]:
                        test_a_i = len(full)
                    elif '1163testb SOCCER' in lines[i]:
                        test_b_i = len(full)

                    full.append([{'words': [], 'mentions': []}])

                else:
                    columns = lines[i].strip('\n').split('\t')

                    if columns[0] == '':
                        full[-1].append({'words': [], 'mentions': []})
                    else:
                        if len(columns) > 1:
                            if columns[1] == 'B':
                                mention = {'text': columns[2],
                                           'start': len(full[-1][-1]['words']),
                                           'end': len(full[-1][-1]['words']),
                                           'tag': columns[3]}

                                if columns[3] != '--NME--' and columns[3] not in self.entity_map:
                                    map_item = {
                                        'wikipedia_url': columns[4],
                                        'wikipedia_id': columns[5]
                                    }

                                    if len(columns) > 6:
                                        map_item['freebase_mid'] = columns[6]

                                    self.entity_map[columns[3]] = map_item

                                full[-1][-1]['mentions'].append(mention)

                            else:
                                full[-1][-1]['mentions'][-1]['end'] = len(full[-1][-1]['words'])

                        full[-1][-1]['words'].append(columns[0])

        filtered = []

        for i in range(len(full)):
            filtered.append([])

            for j in range(len(full[i])):
                if 0 < len(full[i][j]['words']) and len(full[i][0]['words'][0]) > 1:
                    filtered[-1].append(full[i][j])

        self.documents_train = filtered[:test_a_i]
        self.documents_test = filtered[test_a_i:]
        self.documents_test_a = filtered[test_a_i: test_b_i]
        self.documents_test_b = filtered[test_b_i:]
        logging.info(f'Done in {timer.step().string}.')
        return self

    def postprocess(self, predictions):
        pass

    def save(self):
        logging.info('Saving AIDA-YAGO2 dataset...')
        json.dump(self.config.dump(), open(self.config.paths.data.store.config, 'w'), indent=4)
        json.dump(self.documents_train, open(self.config.paths.data.store.train, 'w'), indent=4)
        json.dump(self.documents_test, open(self.config.paths.data.store.test, 'w'), indent=4)
        json.dump(self.documents_test_a, open(self.config.paths.data.store.test_a, 'w'), indent=4)
        json.dump(self.documents_test_b, open(self.config.paths.data.store.test_b, 'w'), indent=4)
        json.dump(self.entity_map, open(self.config.paths.data.store.entity_map, 'w'), indent=4)
        json.dump(self.labels.tags, open(self.config.paths.data.store.labels, 'w'), indent=4)
        logging.info('Done.')
        return self

    def load(self):
        if check_paths(self.config.paths.data.store, ['train', 'entity_map']):
            logging.info('Loading AIDA-YAGO2 dataset...')
            self.documents_train = json.load(open(self.config.paths.data.store.train))
            self.documents_test = json.load(open(self.config.paths.data.store.test))
            self.documents_test_a = json.load(open(self.config.paths.data.store.test_a))
            self.documents_test_b = json.load(open(self.config.paths.data.store.test_b))
            self.entity_map = json.load(open(self.config.paths.data.store.entity_map))
            self.labels.load(json.load(open(self.config.paths.data.store.labels)))
            logging.info('Done.')
            return True

        return False

    def clean(self, full_clean=False, **kwargs):
        shutil.rmtree(self.config.paths.data.store.root)

        if not full_clean:
            self.config.paths.mkdirs(self.melbe_config.paths, **kwargs)

        logging.info('Cleaned existing data.')
        return self

    def retrieve(self):
        if self.config.paths.data.static.url is not None:
            logging.info('Downloading AIDA-YAGO2 dataset...')
            urllib.request.urlretrieve(self.config.paths.data.static.url,
                                       os.path.join(self.config.paths.data.static.root, 'download.zip'))

            logging.info('Extracting AIDA-YAGO2 dataset...')
            with zipfile.ZipFile(os.path.join(self.config.paths.data.static.root, 'download.zip'), 'r') as file:
                file.extractall(self.config.paths.data.static.root)
                os.remove(os.path.join(self.config.paths.data.static.root, 'download.zip'))

            logging.info('Done.')
            return True

        return False
