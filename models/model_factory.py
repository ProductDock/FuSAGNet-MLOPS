from typing import Type, Dict, Any
from models.base_model import BaseModel
from models.fusagnet_model import FuSAGNetModel

class ModelFactory:
    _REGISTRY: Dict[str, Type[BaseModel]] = {
        "fusagnet": FuSAGNetModel,
    }

    @classmethod
    def register_model(cls, name: str, model_cls: Type[BaseModel]):
        cls._REGISTRY[name] = model_cls

    @classmethod
    def create(cls, name: str, model_instance="None", **kwargs) -> BaseModel:
        if name not in cls._REGISTRY:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(cls._REGISTRY.keys())}"
            )
        model_cls = cls._REGISTRY[name]
        return model_cls(model_instance=model_instance, **kwargs)