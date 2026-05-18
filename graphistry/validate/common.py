from typing import Tuple

from graphistry.models.types import ValidationMode, ValidationParam


def normalize_validation_params(
    validate: ValidationParam = "autofix",
    warn: bool = True,
) -> Tuple[ValidationMode, bool]:
    if validate is True:
        validate_mode: ValidationMode = "strict"
    elif validate is False:
        validate_mode = "autofix"
        warn = False
    else:
        validate_mode = validate
    return validate_mode, warn
