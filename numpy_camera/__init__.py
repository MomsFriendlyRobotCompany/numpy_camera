# The MIT License
#
# Copyright (c) 2020 Kevin Walchko

try:
    from importlib.metadata import version # type: ignore
except ImportError:
    from importlib_metadata import version # type: ignore

from .pinhole_camera import PinholeCamera

__license__ = 'MIT'
__author__ = 'Kevin Walchko'
__version__ = version("numpy_camera")
