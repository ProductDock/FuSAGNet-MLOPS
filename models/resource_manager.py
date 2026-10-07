import os
import logging
from dvc.api import DVCFileSystem

from models.model_factory import ModelFactory
from models.model_backend_loader import ModelBackendLoader
from models.base_model import BaseModel
from util.data_manager import get_data_manager

logger = logging.getLogger("ResourceManager")


class ResourceManager:

    def __init__(self, config):
        self.active_model = None
        self.is_fallback: bool = False
        self.config = config

    def initialize_resources(self):
        data_manager = get_data_manager()
        rev = data_manager.git_rev
        local_path = data_manager.local_path
        if local_path and rev:
            raise KeyError("Can not set load_model_path and git_revision at the same time")
        if local_path:
            logger.info("Load from local path")
            self.active_model = self.load_model_from_local()
            if not self.active_model:
                self.is_fallback = True                     
        if rev:    
            logger.info("Load from revision")
            self.active_model = self.load_model_from_dvc()
            if not self.active_model:
                self.is_fallback = True
        return self

    def load_model_from_dvc(self):
        try:
            data_manager = get_data_manager()
            rev = data_manager.git_rev
            logger.info(f"Fetching model revision: {rev}")
            saved_model_path = data_manager.save_path
            ext = os.path.splitext(saved_model_path)[1].lower()
            logger.info(f"Loading model format extension: '{ext}'")
            tmp_model_path = "model.pt"
            fs = DVCFileSystem(repo=".", rev=rev, subrepos=True)
            logger.info(f"Saved model path is {saved_model_path}")
            fs.get("models/fusagnet.pt", "model.pt")
            logger.info("Model is saved")
            self.active_model  = ModelBackendLoader.load_from_file(tmp_model_path)
            logger.info("Model is loaded")
            os.remove(tmp_model_path)
            return self.active_model 

        except Exception as e:
            logger.warning(f"Failed to load DVC revision {rev}: {e}")
            return None

    def load_model_from_local(self):
            try:
                data_manager = get_data_manager()
                load_model_path = data_manager.local_path
                logger.info(f"Fetching model local path: {load_model_path}")
                self.active_model  = ModelBackendLoader.load_from_file(load_model_path)
                logger.info("Model is loaded")
                return self.active_model 
    
            except Exception as e:
                logger.warning(f"Failed to load from local path {load_model_path}: {e}")
                return None

    def fetch_predictions_from_dvc(self):
        data_manager = get_data_manager()
        rev = data_manager.git_rev
        logger.info(f"Syncing predictions from DVC: {rev}")
        fs = DVCFileSystem(repo=".", rev=rev)
        target_dir = self.config.result_path
        fs.get(os.path.join(self.config.result_path, "y_hat"), target_dir, recursive=True)

    def get_model(self) -> BaseModel:
        """Returns a unified BaseModel ready for train or inference."""
        data_manager = get_data_manager()
        model_name = data_manager.model_name
        data_manager = get_data_manager()
        model_specs = data_manager.get_model_specs()
        if self.is_fallback or self.active_model is None:
            logger.info("Creating a new model")
            return ModelFactory.create(name=model_name, model_instance=None, config=self.config, **model_specs)
        
        return ModelFactory.create(name=model_name, model_instance=self.active_model, config=self.config, **model_specs)
