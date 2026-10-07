from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModel(ABC):
    def __init__(self, model_instance: Optional[Any] = None):
        self.model = model_instance

    def train(self, train_data: Any, val_data: Optional[Any] = None, **kwargs):
       pass

    @abstractmethod
    def test(self, X: Any, **kwargs):
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        pass