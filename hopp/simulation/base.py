from abc import ABC, abstractmethod
from typing import Any, Dict, Final

import attrs

from hopp.type_dec import FromDictMixin
from hopp.logging_manager import LoggerBase

class BaseClass(LoggerBase, FromDictMixin):
    """
    BaseClass object class. This class does the logging and MixIn class inheritance.
    """

    @classmethod
    def get_model_defaults(cls) -> Dict[str, Any]:
        """Produces a dictionary of the keyword arguments and their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary of keyword argument: default.
        """
        return {el.name: el.default for el in attrs.fields(cls)}

    def _get_model_dict(self) -> dict:
        """Convenience method that wraps the `attrs.asdict` method. Returns the object's
        parameters as a dictionary.

        Returns
        -------
        dict
            The provided or default, if no input provided, model settings as a dictionary.
        """
        return attrs.asdict(self)

class BaseModel(BaseClass, ABC):
    """
    BaseModel is the generic class for any HOPP models. It defines the API required to
    create a valid model.
    """

    NUM_EPS: Final[float] = 0.001  # This is a numerical epsilon to prevent divide by zeros

    @abstractmethod
    def process_inputs() -> dict:
        raise NotImplementedError("BaseModel.process_inputs")

    @abstractmethod
    def process_outputs() -> dict:
        raise NotImplementedError("BaseModel.process_outputs")

class SystemModel(BaseModel, ABC):
    """
    """

    @abstractmethod
    def simulate_system_model() -> None:
        raise NotImplementedError("BaseModel.simulate_system_model")

class FinancialModel(BaseModel, ABC):
    """
    """

    @abstractmethod
    def simulate_financial_model() -> None:
        raise NotImplementedError("BaseModel.simulate_financial_model")

class ControlModel(BaseModel, ABC):
    """
    """

    @abstractmethod
    def simulate_control_model() -> None:
        raise NotImplementedError("BaseModel.simulate_control_model")
