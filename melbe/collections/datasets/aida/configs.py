import os

from melbe.configs import Config, PathConfig, MelbeConfig, MelbePathConfig, init, mkdir, select
from melbe.data import DatasetConfig


class AidaPathConfig(PathConfig):
    name = 'paths'

    class Data(Config):
        name = 'data'

        class Static(Config):
            name = 'static'

            def __init__(self, config: MelbePathConfig, **kwargs):
                super().__init__(**init(kwargs))
                self.root = mkdir(os.path.join(config.data.static.datasets, 'aida'))
                self.all = os.path.join(self.root, 'AIDA-YAGO2-dataset.tsv')
                self.url = os.environ.get('MELBEIN_DATA_DATASETS_AIDA_URL', None)

        class Store(Config):
            name = 'store'

            def __init__(self, config: MelbePathConfig, **kwargs):
                super().__init__(**init(kwargs))
                self.root = mkdir(os.path.join(config.data.store.datasets, 'aida'))
                self.root = mkdir(os.path.join(self.root, f'v-{AidaConfig.version}'))
                self.config = os.path.join(self.root, 'config.json')
                self.train = os.path.join(self.root, 'train.json')
                self.test = os.path.join(self.root, 'test.json')
                self.test_a = os.path.join(self.root, 'test-a.json')
                self.test_b = os.path.join(self.root, 'test-b.json')
                self.labels = os.path.join(self.root, 'labels.json')
                self.entity_map = os.path.join(self.root, 'entity-map.json')

        def __init__(self, config: MelbePathConfig, **kwargs):
            super().__init__(**init(kwargs))
            self.static = AidaPathConfig.Data.Static(config, **select(kwargs, AidaPathConfig.Data.Static.name))
            self.store = AidaPathConfig.Data.Store(config, **select(kwargs, AidaPathConfig.Data.Store.name))

    def __init__(self, config: MelbePathConfig, **kwargs):
        super().__init__(**init(kwargs))
        self.data = AidaPathConfig.Data(config, **select(kwargs, AidaPathConfig.Data.name))

    def mkdirs(self, config: MelbePathConfig, **kwargs):
        self.data = AidaPathConfig.Data(config, **select(kwargs, AidaPathConfig.Data.name))


class AidaConfig(DatasetConfig):
    version = '0.1.0'
    name = 'aida'

    def __init__(self, config: MelbeConfig, **kwargs):
        super().__init__(**init(kwargs))
        self.paths = AidaPathConfig(config.paths, **select(kwargs, AidaPathConfig.name))
