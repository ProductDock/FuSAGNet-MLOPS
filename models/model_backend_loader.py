import os
import logging
import joblib
import torch
from models.FuSAGNet import FuSAGNet
from util.data_manager import get_data_manager

logger = logging.getLogger("BackendLoader")

class ModelBackendLoader:
    """Deserializes raw framework models based on file type or extension."""

    @staticmethod
    def load_from_file(local_path: str, **kwargs):
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file asset missing at: {local_path}")

        ext = os.path.splitext(local_path)[1].lower()
        logger.info(f"Loading model format extension: '{ext}'")

        if ext in [".pkl", ".joblib"]:
            return joblib.load(local_path)
        elif ext in [".pt"]:
            model = kwargs.get('model') 
            data_manager = get_data_manager()
            try:
                model = FuSAGNet(
                edge_index_sets=[data_manager.fc_edge_index],
                node_num=data_manager.node_num, 
                dim=data_manager.train_config["dim"], 
                window_size=data_manager.window_size ,
                out_layer_num= data_manager.out_layer_num,
                out_layer_inter_dim=data_manager.out_layer_inter_dim,
                topk=data_manager.topk,
                process_dict=data_manager.process_dict,
                )              
            except Exception as e:
                logger.warning(f"Failed to create model: {e}")
                return None
            device = data_manager.device
            model.load_state_dict(torch.load(local_path, map_location=device))
            return model 
        else:
            raise ValueError(f"Extension '{ext}' is not supported.")
        