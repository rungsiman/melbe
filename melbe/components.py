import importlib
from typing import Any, Dict, List, Type

from melbe.configs import MelbeConfig


def init_component(config: MelbeConfig,
                   global_kwargs: Dict[str, Any],
                   parents: List[str] = None,
                   branch: str = None,
                   custom_cls: Type = None,
                   **kwargs):

    if custom_cls is not None:
        return custom_cls(config, **{**global_kwargs, **kwargs})

    cls, module = None, None

    if '/' in branch or '.' in branch:
        if '/' in branch:
            module_name = branch.split('/')[0] + '.' + branch.split('/')[0]
            cls_name = branch.split('/')[1]

        else:
            module_name = '.'.join(branch.split('.')[:-1])
            cls_name = branch.split('.')[-1]

        for parent in parents:
            try:
                module = importlib.import_module(f'{parent}.{module_name}')
            except ModuleNotFoundError:
                continue

        if module is None:
            raise ModuleNotFoundError(f'No modules found for "{branch}" in "{parents}".')

        cls = getattr(module, cls_name)

    else:
        for parent in parents:
            try:
                module = importlib.import_module(f'{parent}.{branch}')
            except ModuleNotFoundError:
                continue

        if module is None:
            raise ModuleNotFoundError(f'No modules found for "{branch}" in "{parents}".')

        for attr in dir(module):
            if attr.lower() == branch.lower():
                cls = getattr(module, attr)

    return cls(config, **{**global_kwargs, **kwargs}) if cls is not None else None
