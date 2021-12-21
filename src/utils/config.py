import sys
import yaml
from typing import Any
from pathlib import Path
from yaml.scanner import ScannerError

from .log import create_logger

log = create_logger(__name__)


class ConfigType:
    '''
    An object-relational mapper (ORM) type class used in `config.Config` to create nested
    object attributes. For more information on Python ORMs, go
    [here](https://www.fullstackpython.com/object-relational-mappers-orms.html).
    '''
    def __init__(self, **kwargs) -> None:
        '''
        All `kwargs` passed in will be nested into attributes if the `kwargs` consist of
        nested dictionaries.
        '''
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = ConfigType(**value)
            setattr(self, key, self._type(value))

    def _type(self, obj: Any) -> Any:
        '''
        Type strings into tuples when setting attributes.
        '''
        if isinstance(obj, str):
            if '(' in obj:
                return tuple(map(int, obj[1:-1].split(', ')))
        else:
            return obj


class Config:
    '''
    Reads in a YAML file and assigns nested attributes to itself based on structure of
    YAML
    '''
    def __init__(self, path: Path) -> None:
        '''
        Arguments:
            path: The path to the `config.yml` file. This file can have any name, but
                  needs to be a valid YAML.

        Returns:
            None
        '''
        self.path = path
        self._load()
        self._set_attributes()

    def _load(self) -> None:
        '''
        Loads the YAML file and assigns to `self.data`
        '''
        try:
            with open(self.path, 'r') as f:
                try:
                    self.data = yaml.safe_load(f)
                except ScannerError as e:
                    log.error(f'Error parsing YAML file at {self.path}:\n' f'{e.error}')
                    sys.exit(1)
        except FileNotFoundError:
            log.error(f'Error opening YAML file at {self.path}. Does it exist?')
            sys.exit(1)

    def _set_attributes(self) -> None:
        '''
        Set all nested attributes using the `config.ConfigType` class
        '''
        for key, value in self.data.items():
            if isinstance(value, dict):
                value = ConfigType(**value)
            setattr(self, key, value)

    def save(self, path: Path) -> None:
        '''
        Save the current config to the specified `path`

        Arguments:
            path: The path and filename of the output YAML file.

        Returns:
            None
        '''
        with open(path, 'w') as f:
            yaml.dump(self.data, f)
