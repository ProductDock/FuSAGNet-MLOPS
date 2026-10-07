import os

import traceback

from flask import Flask, json, jsonify, request
import pandas as pd

import logging
from models.resource_manager import ResourceManager
from util.data_manager import initialize_data_manager

logger = logging.getLogger("service")
GET_MODEL = os.getenv('APP_ROOT', '/model')
TEST_ROOT = os.getenv('APP_ROOT', '/')
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 5000))
SAVED_MODEL_PATH = os.getenv('MODEL_PATH', 'models/fusagnet.pt')

train_config = {
        "batch": 128,
        "epoch": 1,
        "slide_win": 3,
        "dim": 16,
        "slide_stride": 5,
        "comment": "",
        "seed": 999,
        "out_layer_num": 1,
        "out_layer_inter_dim": 64,
        "decay": 0,
        "val_ratio": 0.2,
        "topk": 15,
        "lr": 1e-3,
        "gpu_id": 1,
        "alpha": 0.5,
        "beta": 1.0
    }

env_config = {
        "save_path": "",
        "dataset": "swat",
        "report": "best",
        "device": "cpu",
        "load_model_path":"models/fusagnet.pt",
        "results_path": "results/",
        "train": False,
        "git_revision": "",
        "model_name": "fusagnet",
        "saved_model_path": SAVED_MODEL_PATH,
    }
app = Flask(__name__)
data_manager = initialize_data_manager(train_config, env_config)
resource_mgr = ResourceManager(train_config)

model = resource_mgr.load_model_from_local()

@app.route(GET_MODEL, methods=["GET"])
def predict_from_disk():
    try:
      
        logger.info("Configuration is loaded")
        logger.info(f"config {model}")
        layers_summary = []

        for name, module in model.named_children():
            layer_info = {
                "layer_name": name,
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                ),
            }
            logger.info(f"Layer Name: {name} | Type: {type(module).__name__}")
            layers_summary.append(layer_info)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

        return {
            "model_name": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture_flow": layers_summary,
        }
            
                    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route(TEST_ROOT, methods=["GET"])
def setup_test():
    print("Hello")
    return 'Hello'

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run(host=HOST, port = PORT_NUMBER)
