from pathlib import Path
import logging
from typing import Optional
import torch
from models.base_model import BaseModel
from models.FuSAGNet import FuSAGNet
from runner.test import test as run_pytorch_test
from runner.train import train as run_pytorch_train
import torch


logger = logging.getLogger("FuSAGNetModel")


class FuSAGNetModel(BaseModel):
    """BaseModel wrapper for the FuSAGNet PyTorch model architecture."""

    def __init__(
        self,
        config,
        edge_index_sets,
        node_num, 
        process_dict,
        model_instance: Optional[FuSAGNet] = None,
    ):
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config["gpu_id"]}')
            torch.cuda.set_device(self.device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if model_instance is None:
            model_instance = FuSAGNet(
                edge_index_sets=edge_index_sets,
                node_num=node_num,
                dim=config["dim"],
                window_size=config["slide_win"],
                out_layer_num=config["out_layer_num"],
                out_layer_inter_dim=config["out_layer_inter_dim"],
                topk=config["topk"],
                process_dict=process_dict,
            )

        model_instance = model_instance.to(self.device)
        super().__init__(model_instance=model_instance)

    def train(
        self,
        train_dataloader,
        val_dataloader,
        save_path: str = "models/fusagnet.pt",
    ):
        logger.info("Model training")

        best_state_dict, loss_history = run_pytorch_train(
            model=self.model,
            save_path=save_path,
            config=self.config,
            train_dataloader= train_dataloader,
            val_dataloader= val_dataloader,
            device=self.device,
        )
        self.model.load_state_dict(best_state_dict)
        self.save_model(save_path)
        return loss_history

    def test(self, dataloader):
        logger.info("Model testing")

        return run_pytorch_test(
            self.model, dataloader, device=self.device, config=self.config
        )

    def save_model(self, filepath: str = "models/fusagnet.pt") -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"FuSAGNet model state saved to {filepath}")

    def load_weights(self, filepath: str) -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        state_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Model state dictionary loaded from {filepath}")