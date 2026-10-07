import argparse
import os
import random
from pathlib import Path
import logging
import numpy as np

import torch


from models.resource_manager import ResourceManager
from util.data_manager import initialize_data_manager
from util.data_manager import get_data_manager

from runner.evaluate import (
    get_best_performance_data,
    get_full_err_scores,
)

SAVED_MODEL_PATH = os.getenv("MODEL_PATH", "models/fusagnet.pt")

logger = logging.getLogger("Main")


class Main:
    def __init__(self, train_config, env_config, resource_mgr):
        self.train_config = train_config
        self.env_config = env_config
        self.resource_mgr = resource_mgr

    def run(self):
        self.model = self.resource_mgr.get_model()
        data_manager = get_data_manager()

        if self.env_config["train"] == True:
            self.create_paths()
            self.train_log = self.model.train(save_path=SAVED_MODEL_PATH, train_dataloader = data_manager.train_dataloader, val_dataloader = data_manager.val_dataloader)

        _, self.test_result = self.model.test(data_manager.test_dataloader)
        _, self.val_result = self.model.test(data_manager.val_dataloader)
    
        self.get_score(
            self.test_result["forecasting"],
            self.test_result["reconstruction"],
            self.val_result["forecasting"],
            self.val_result["reconstruction"],
            self.train_config,
            self.env_config["dataset"],
        )

    def get_score(
        self,
        test_result_f,
        test_result_r,
        val_result_f,
        val_result_r,
        config,
        dataset_name,
    ):
        def whm(x1, x2, w1, w2):
            epsilon = 1e-2
            return (w1 + w2) * (x1 * x2) / (w1 * x2 + w2 * x1 + epsilon)

        _, _, test_labels_f = test_result_f
        test_labels = np.asarray(test_labels_f)[:, 0].tolist()

        alpha = config["alpha"]
        test_scores_f, _ = get_full_err_scores(test_result_f[:2], val_result_f[:2])
        test_scores_r, _ = get_full_err_scores(test_result_r[:2], val_result_r[:2])
        test_scores_whm = []
        for i in range(len(test_scores_f)):
            score = whm(
                x1=test_scores_f[i], x2=test_scores_r[i], w1=alpha, w2=1.0 - alpha
            )
            test_scores_whm.append(score)

        results_config = self.env_config["results_path"]
        results_save_path = f"{results_config}/{dataset_name}/"
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path, exist_ok=True)
        all_scores = [test_scores_f, test_scores_r, test_scores_whm]
        score_labels = ["Forecasting", "Reconstruction", "Weighted Harmonic Mean"]
        to_save = {}
        if self.env_config["report"] == "best":
            for i, scores in enumerate(all_scores):
                score_label = score_labels[i]
                if score_label != "Weighted Harmonic Mean":
                    continue

                scores = np.array(scores)
                to_save[score_label] = scores
                top1_best_info = get_best_performance_data(scores, test_labels, topk=1)
                print_score =  f"F1: {top1_best_info[0]:.4f} | Pr: {top1_best_info[1]:.4f} | Re: {top1_best_info[2]:.4f}"
                print(
                   print_score
                )
                with open(f"{results_config}/results.txt", "a", encoding="utf-8") as file:
                    file.write(print_score)

        for k in to_save:
            f2save = to_save.get(k)
            np.save(os.path.join(results_save_path, k), f2save)

    def create_paths(self):
        dataset = self.env_config["dataset"]
        resulth_path = self.env_config["results_path"]

        results_config = self.env_config["results_path"]
        resulth_path = f"{results_config}/{dataset}/"
        paths = [ SAVED_MODEL_PATH,
                resulth_path,
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch", help="batch size", type=int, default=128)
    parser.add_argument("-epoch", help="train epoch", type=int, default=1)
    parser.add_argument("-slide_win", help="window size", type=int, default=3)
    parser.add_argument("-dim", help="dimension", type=int, default=16)
    parser.add_argument("-slide_stride", help="window stride", type=int, default=5)
    parser.add_argument("-save_path_pattern", help="save path", type=str, default="")
    parser.add_argument("-dataset", help="hai/swat/wadi", type=str, default="swat")
    parser.add_argument("-device", help="cpu/cuda", type=str, default="cpu")
    parser.add_argument("-random_seed", help="random seed", type=int, default=-999)
    parser.add_argument("-comment", help="experiment comment", type=str, default="")
    parser.add_argument(
        "-out_layer_num", help="out layer dimension", type=int, default=1
    )
    parser.add_argument(
        "-out_layer_inter_dim",
        help="intermediate out layer dimension",
        type=int,
        default=64,
    )
    parser.add_argument("-decay", help="weight decay", type=float, default=0)
    parser.add_argument(
        "-val_ratio", help="validation data ratio", type=float, default=0.2
    )
    parser.add_argument("-topk", help="k", type=int, default=15)
    parser.add_argument("-report", help="best/val", type=str, default="best")
    parser.add_argument(
        "-load_model_path", help="trained model path", type=str, default=""
    )
    # models/results/fusagnet.pt 
    parser.add_argument("-lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-gpu_id", help="gpu device ID", type=int, default=1)
    parser.add_argument(
        "-alpha", help="forecasting loss weight", type=float, default=0.5
    )
    parser.add_argument("-beta", help="sparse loss weight", type=float, default=1.0)
    parser.add_argument("-results_path", help="results path", type=str, default="results/")
    parser.add_argument("-train", help="train new model or use exisitng one", type=bool, default=True)
    parser.add_argument("-git_revision", help= "revision to pull the model from", type=str, default="") 
    # fusagnet#stage#version
    parser.add_argument("-model_name", help= "model name", type=str, default='fusagnet')

    args = parser.parse_args()

    if args.random_seed < 0:
        args.random_seed = random.randint(0, 100)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    train_config = {
        "batch": args.batch,
        "epoch": args.epoch,
        "slide_win": args.slide_win,
        "dim": args.dim,
        "slide_stride": args.slide_stride,
        "comment": args.comment,
        "seed": args.random_seed,
        "out_layer_num": args.out_layer_num,
        "out_layer_inter_dim": args.out_layer_inter_dim,
        "decay": args.decay,
        "val_ratio": args.val_ratio,
        "topk": args.topk,
        "lr": args.lr,
        "gpu_id": args.gpu_id,
        "alpha": args.alpha,
        "beta": args.beta
    }

    env_config = {
        "save_path": args.save_path_pattern,
        "dataset": args.dataset,
        "report": args.report,
        "device": args.device,
        "load_model_path": args.load_model_path,
        "results_path": args.results_path,
        "train": args.train,
        "git_revision": args.git_revision,
        "model_name": args.model_name,
        "saved_model_path": SAVED_MODEL_PATH,
    }

    data_manager = initialize_data_manager(train_config, env_config)
    resource_mgr = ResourceManager(train_config)
    resource_mgr = resource_mgr.initialize_resources()
    main = Main(train_config = train_config, env_config= env_config, resource_mgr=resource_mgr)
  

    main.run()
