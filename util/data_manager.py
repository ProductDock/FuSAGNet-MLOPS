# util/data_manager.py

import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import random
import logging

from datasets.TimeDataset import TimeDataset
from util.net_struct import get_fc_graph_struc, get_feature_map
from util.preprocess import build_loc_net, construct_data

DATA_MANAGER_INSTANCE = None

class DataManager:
    def __init__(self, train_config, env_config):
        self.train_config = train_config
        self.env_config = env_config
        self.dataset_name = env_config["dataset"]

        self.feature_map = None
        self.fc_edge_index = None
        self.process_dict = None
        self.node_num = None
        self.out_layer_inter_dim=train_config["out_layer_inter_dim"]
        self.out_layer_num=train_config["out_layer_num"]
        self.window_size=train_config["slide_win"]
        self.topk = train_config["topk"]

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.git_rev = env_config["git_revision"]
        self.local_path = env_config["load_model_path"]
        self.save_path = env_config["saved_model_path"]
        self.model_name = env_config["model_name"]
        self.device = env_config["device"]

        self.prepare_data()
        self.setup_logging()

    def prepare_data(self):
        dataset = self.dataset_name
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test_100.csv", sep=",", index_col=0)

        if dataset in ["swat", "wadi"]:
            train, test = train_orig[2160:], test_orig
        else:
            train, test = train_orig, test_orig

        if "attack" in train.columns:
            train = train.drop(columns=["attack"])

        self.feature_map = get_feature_map(dataset)
        self.node_num = len(self.feature_map)
        fc_struc = get_fc_graph_struc(dataset)

        # Build feature graph edge indices
        edge_index = build_loc_net(
            fc_struc, list(train.columns), feature_map=self.feature_map
        )
        self.fc_edge_index = torch.tensor(edge_index, dtype=torch.long)

        train_dataset_indata = construct_data(train, self.feature_map, labels=0)
        test_dataset_indata = construct_data(
            test, self.feature_map, labels=test.attack.tolist()
        )

        cfg = {
            "slide_win": self.train_config["slide_win"],
            "slide_stride": self.train_config["slide_stride"],
        }

        train_dataset = TimeDataset(
            train_dataset_indata,
            self.fc_edge_index,
            mode="train",
            task="forecasting",
            config=cfg,
        )
        min_train, max_train = train_dataset.get_train_min_max()
        test_dataset = TimeDataset(
            test_dataset_indata,
            self.fc_edge_index,
            mode="test",
            task="forecasting",
            min_train=min_train,
            max_train=max_train,
            config=cfg,
        )

        self.train_dataloader, self.val_dataloader = self.get_train_val_loaders(
            train_dataset, self.train_config["batch"], val_ratio=self.train_config["val_ratio"]
        )

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.train_config["batch"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        self.process_dict = self.get_process_dict(dataset)

    def get_train_val_loaders(self, train_dataset, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len :]]
        )
        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]

        train_dataloader = DataLoader(
            Subset(train_dataset, train_sub_indices),
            batch_size=batch,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_dataloader = DataLoader(
            Subset(train_dataset, val_sub_indices),
            batch_size=batch,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        return train_dataloader, val_dataloader

    @staticmethod
    def get_process_dict(dataset_name: str) -> dict:
        mapping = {
            "hai": {"P1": 38, "P2": 22, "P3": 7, "P4": 12},
            "swat": {"P1": 5, "P2": 11, "P3": 9, "P4": 9, "P5": 13, "P6": 4},
            "wadi": {"P1": 19, "P2": 90, "P3": 15, "P4": 3},
        }
        return mapping.get(dataset_name, {})

    def get_model_specs(self) -> dict:
        return {
            "edge_index_sets": [self.fc_edge_index],
            "node_num": len(self.feature_map),
            "process_dict": self.process_dict
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True)
        
def initialize_data_manager(train_config: dict, env_config: dict) -> DataManager:
    global DATA_MANAGER_INSTANCE
    DATA_MANAGER_INSTANCE = DataManager(train_config, env_config)
    return DATA_MANAGER_INSTANCE

def get_data_manager() -> DataManager:
    if DATA_MANAGER_INSTANCE is None:
        raise RuntimeError(
            "DataManager has not been initialized. Call initialize_data_manager(...) at application start."
        )
    return DATA_MANAGER_INSTANCE