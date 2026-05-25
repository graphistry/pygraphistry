"""
Type definitions for data validation modes.

Used by plot()/upload() validation parameters and arrow uploader behavior.
"""

from typing import Literal, Union


ValidationMode = Literal['strict', 'strict-fast', 'autofix']
ValidationParam = Union[ValidationMode, bool]
SchemaValidationMode = Literal['strict', 'autofix']
SchemaValidationParam = Union[SchemaValidationMode, bool]
