from abc import ABC
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from graphistry.utils.json import JSONVal, serialize_to_json_val

if TYPE_CHECKING:
    from graphistry.compute.exceptions import GFQLValidationError


class ASTSerializable(ABC):
    """
    Internal, not intended for use outside of this module.
    Class name becomes o['type'], and all non reserved_fields become JSON-typed key
    """

    reserved_fields = ['type']

    def validate(self, collect_all: bool = False) -> Optional[List['GFQLValidationError']]:
        """Validate this AST node.

        Args:
            collect_all: If True, collect all errors instead of raising on first.
                        If False (default), raise on first error.

        Returns:
            If collect_all=True: List of validation errors (empty if valid)
            If collect_all=False: None if valid

        Raises:
            GFQLValidationError: If collect_all=False and validation fails
        """
        if not collect_all:
            # Fail fast mode - raise on first error
            self._validate_fields()
            # Validate children
            for child in self._get_child_validators():
                child.validate(collect_all=False)
            return None

        # Collect all errors mode
        errors: List['GFQLValidationError'] = []

        # Collect own validation errors
        try:
            self._validate_fields()
        except Exception as e:
            # Import here to avoid circular dependency
            from graphistry.compute.exceptions import GFQLValidationError
            if isinstance(e, GFQLValidationError):
                errors.append(e)
            else:
                # Re-raise non-validation errors
                raise

        # Collect child validation errors
        for child in self._get_child_validators():
            child_errors = child.validate(collect_all=True)
            if child_errors:
                errors.extend(child_errors)

        return errors

    def _validate_fields(self) -> None:
        """Override in subclasses to validate specific fields.

        Should raise GFQLValidationError for validation failures.
        Default implementation does nothing (for backward compatibility).
        """
        pass

    def _get_child_validators(self) -> Sequence['ASTSerializable']:
        """Override in subclasses to return child AST nodes that need validation.

        Returns:
            Sequence of child AST nodes to validate
        """
        return []

    def to_json(self, validate=True) -> Dict[str, JSONVal]:
        """
        Returns JSON-compatible dictionry {"type": "ClassName", "arg1": val1, ...}
        Emits all non-reserved instance fields
        """
        if validate:
            self.validate()
        data: Dict[str, JSONVal] = {'type': self.__class__.__name__}
        for key, value in self.__dict__.items():
            if key not in self.reserved_fields:
                data[key] = serialize_to_json_val(value)
        return data

    @classmethod
    def from_json(cls, d: Dict[str, JSONVal], validate: bool = True) -> 'ASTSerializable':
        """
        Given c.to_json(), hydrate back c

        Args:
            d: Dictionary from to_json()
            validate: If True (default), validate after parsing

        Returns:
            Hydrated AST object

        Raises:
            GFQLValidationError: If validate=True and validation fails
        """
        constructor_args = {k: v for k, v in d.items() if k not in cls.reserved_fields}
        instance = cls(**constructor_args)

        if validate:
            instance.validate(collect_all=False)

        return instance
