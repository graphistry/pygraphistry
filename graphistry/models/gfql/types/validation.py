"""
Type definitions for data validation modes.

Used by plot()/upload() validation parameters and arrow uploader behavior.
"""

from typing import Literal, Union


ValidationMode = Literal['strict', 'strict-fast', 'autofix']
ValidationModeOrNone = Union[ValidationMode, Literal['none']]
ValidationParam = Union[ValidationMode, bool]
ValidationParamOrNone = Union[ValidationModeOrNone, bool]
