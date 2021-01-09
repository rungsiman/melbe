import importlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Type, Union


class Config:
    """Template for configuration classes

    The attributes of a configuration class can be set directly from command arguments using function
    "select" defined in this module. For example, suppose class "FooConfig" has an attribute "bar"
    and is passed kwargs during initialization, running "bin/train --bar=hello" will set the attribute
    to "hello."

    Call the class' method "describe" to get all configurations.
    """

    _melbe_config: bool = True
    _melbe_obj: bool = False
    name: str = ...

    def __init__(self, **kwargs):
        self._melbe_obj = True

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _attrs(config):
        if getattr(config, '_melbe_meta', False) in ('dataset', 'pipeline'):
            return Config._attrs(getattr(config, 'config'))

        elif str(config).startswith('<function'):
            return str(config)

        elif str(config).startswith('<class') and not getattr(config, '_melbe_obj', False):
            return str(config)

        elif any([isinstance(config, t) for t in [str, bool, int, float]]) or config is None:
            return config

        elif isinstance(config, list):
            return [Config._attrs(item) for item in config]

        elif isinstance(config, dict):
            return {key: Config._attrs(value) for key, value in config.items()}

        else:
            descriptions = {}

            for attr in dir(config):
                if attr not in ('_attrs', 'attr', 'describe', 'dump', 'assign', 'override', 'mkdirs') \
                        and not attr.startswith('__') and not attr.startswith('_melbe') \
                        and (not getattr(getattr(config, attr), '_melbe_config', False) or
                             getattr(getattr(config, attr), '_melbe_obj', False)) \
                        and not (getattr(config, '_melbe_class_config', False) and attr == 'cls'):
                    descriptions[attr] = Config._attrs(getattr(config, attr))

            return descriptions

    def describe(self) -> str:
        return json.dumps(Config._attrs(self), indent=4)

    def dump(self) -> object:
        return Config._attrs(self)

    def attr(self, key: str):
        return getattr(self, key)


class ClassConfig(Config):
    _melbe_class_config: bool = True

    cls: Union[Any, None] = ...
    module: Union[str, None] = ...
    name: Union[str, None] = ...
    kwargs: Dict[str, Any] = ...

    def __init__(self, **kwargs):
        super().__init__(**init(kwargs))
        self.kwargs = {**select(kwargs, 'kwargs')}

        if isinstance(self.module, str) and isinstance(self.name, str):
            self.cls = getattr(importlib.import_module(self.module), self.name)
        else:
            self.cls = None

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

    def assign(self, module: str, name: str):
        self.module = module
        self.name = name
        self.cls = getattr(importlib.import_module(self.module), self.name)

    def override(self, cls: Any):
        self.cls = cls
        self.module = None
        self.name = None


class MelbeConfig(Config):
    """Melbe project's default configurations"""
    version: str = '0.1.0'
    name: str = 'melbe'

    def __init__(self, **kwargs):
        super().__init__(**init(kwargs))
        self.paths = MelbePathConfig(**select(kwargs, MelbePathConfig.name))


class MelbePathConfig(Config):
    name: str = 'paths'

    class Data(Config):
        name: str = 'data'

        class Logs(Config):
            name: str = 'logs'

            def __init__(self, parent: str, **kwargs):
                super().__init__(**init(kwargs))
                self.root = mkdir(os.path.join(parent, 'logs'))
                self.pipelines = mkdir(os.path.join(self.root, 'pipelines'))

        class Secrets(Config):
            name: str = 'secrets'

            def __init__(self, parent: str, **kwargs):
                super().__init__(**init(kwargs))
                self.root = mkdir(os.path.join(parent, 'secrets'))
                self.datasets = mkdir(os.path.join(self.root, 'datasets'))

        class Static(Config):
            name: str = 'static'

            def __init__(self, parent: str, **kwargs):
                super().__init__(**init(kwargs))
                self.root = mkdir(os.path.join(parent, 'static'))
                self.datasets = mkdir(os.path.join(self.root, 'datasets'))

        class Store(Config):
            name: str = 'store'

            def __init__(self, parent: str, **kwargs):
                super().__init__(**init(kwargs))
                self.root = mkdir(os.path.join(parent, 'store'))
                self.datasets = mkdir(os.path.join(self.root, 'datasets'))
                self.pipelines = mkdir(os.path.join(self.root, 'pipelines'))

        def __init__(self, **kwargs):
            super().__init__(**init(kwargs))
            self.root = mkdir(getattr(self, 'root', 'data'))
            self.logs = MelbePathConfig.Data.Logs(self.root, **select(kwargs, MelbePathConfig.Data.Logs.name))
            self.secrets = MelbePathConfig.Data.Secrets(self.root, **select(kwargs, MelbePathConfig.Data.Secrets.name))
            self.static = MelbePathConfig.Data.Static(self.root, **select(kwargs, MelbePathConfig.Data.Static.name))
            self.store = MelbePathConfig.Data.Store(self.root, **select(kwargs, MelbePathConfig.Data.Store.name))

    class Components(Config):
        name: str = 'components'

        def __init__(self, **kwargs):
            super().__init__(**init(kwargs))
            self.root = 'melbein'
            self.datasets = f'{self.root}.datasets'
            self.pipelines = f'{self.root}.pipelines'

    def __init__(self, **kwargs):
        super().__init__(**init(kwargs))
        self.data = MelbePathConfig.Data(**select(kwargs, MelbePathConfig.Data.name))
        self.components = MelbePathConfig.Components(**select(kwargs, MelbePathConfig.Components.name))


class PathConfig(Config):
    def mkdirs(self, config: MelbePathConfig, **kwargs):
        ...


def init(kwargs: Dict[str, str]):
    """Filter kwargs that do not belong to subclasses"""

    return {key: value for key, value in kwargs.items() if '-' not in key}


def select(kwargs: Dict[str, str], *tags: str, reverse: bool = False) -> Dict[str, str]:
    """Filter kwargs using tags

    Suppose a dictionary "kwargs" contains {'a': 1, 'b': 2, 'c': 3}, using "select(kwargs, 'b', 'c')" or
    "select(kwargs, 'a', reverse=True)" will return a dictionary {'b': 2, 'c': 3}.
    """

    if reverse:
        return {key: value for key, value in kwargs.items() if all(not key.startswith(cls) for cls in tags)}
    else:
        filtered_kwargs = {}

        for tag in tags:
            for key, value in kwargs.items():
                if key.startswith(tag):
                    filtered_kwargs[re.sub(rf'^{tag}-', '', key)] = value

        return filtered_kwargs


def mkdir(path: str) -> str:
    """Create a directory if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def parse_kwargs(kwargs_list: List[str]) -> Dict[str, Any]:
    """Parse string-based commandline kwargs into a dictionary"""

    kwargs_dict = {}

    for kwarg in kwargs_list:
        key = kwarg[2:].split('=')[0]
        value = '='.join(kwarg.split('=')[1:])

        try:
            if re.match(r'^(-)?[0-9]+$', value):
                value = int(value)

            elif re.match(r'^(-)?[0-9]*.[0-9]+$', value) or re.match(r'^(-)?[0-9]*(\.)?[0-9]+e(-|\+)[0-9]+$', value):
                value = float(value)

            elif re.match(r'^\[.*]$', value) or re.match(r'^\{.*}$', value):
                value = json.loads(value)

            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'

            elif value.lower() == 'none':
                value = None

        except:
            logging.warning(f'Could not automatically parse argument "{key}." Its type will remain string.')

        kwargs_dict[key] = value

    return kwargs_dict


def check_paths(config: Config, attrs: List[str]) -> bool:
    return all(os.path.exists(getattr(config, attr)) for attr in attrs)
