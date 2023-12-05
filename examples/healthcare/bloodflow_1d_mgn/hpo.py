import torch
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch
import hydra
from omegaconf import DictConfig
from train import do_training
import os
from modulus.distributed.manager import DistributedManager
import random
import numpy as np

# initialize distributed manager
DistributedManager.initialize()
dist = DistributedManager()

def objective(config, cfg):  
    cfg.checkpoints.ckpt_path = os.getcwd() + "/" + cfg.checkpoints.ckpt_path 
    print("cfg.checkpoints.ckpt_path", cfg.checkpoints.ckpt_path)

    cfg.scheduler.lr = config["lr"] 
    cfg.scheduler.lr_decay = config["lr_decay"]
    cfg.training.batch_size = config["batch_size"]
    cfg.training.stride = config["stride"]
    cfg.training.rate_noise = config["rate_noise"]
    cfg.training.loss_weight_boundary_nodes = config["loss_weight_boundary_nodes"]
    cfg.architecture.processor_size = config["processor_size"]
    cfg.architecture.hidden_dim_node_encoder = config["hidden_dim_node_encoder"]
    cfg.architecture.hidden_dim_edge_encoder = config["hidden_dim_edge_encoder"]
    cfg.architecture.hidden_dim_processor = config["hidden_dim_processor"]
    cfg.architecture.hidden_dim_node_decoder = config["hidden_dim_node_decoder"]

    metric = do_training(cfg).cpu().detach().numpy()
    
    train.report({"inference_performance": float(metric)})  # Report to Tune

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    cfg.work_directory = os.getcwd()
    print("cfg.work_directory", cfg.work_directory )
    cfg.training.output_interval = cfg.training.epochs - 1 

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "lr_decay": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.randint(10, 100), 
        "stride": tune.randint(5, 10), 
        "rate_noise": tune.randint(1, 200),
        "loss_weight_boundary_nodes": tune.randint(1, 200), 
        "processor_size": tune.randint(0, 10),
        "hidden_dim_node_encoder": tune.randint(1, 100),
        "hidden_dim_edge_encoder": tune.randint(1, 100),
        "hidden_dim_processor": tune.randint(1, 100),
        "hidden_dim_node_decoder": tune.randint(1, 100),
    }
    algo = OptunaSearch()  

    def objective_cfg(config):

        return objective(config, cfg)

    objective_with_gpu = tune.with_resources(objective_cfg, {"gpu": 1})

    storage_path = os.path.expanduser("/home/aiacovelli/ray_results")
    exp_name = "hpo_MeshGraphNet"
    path = os.path.join(storage_path, exp_name)

    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(
            path, 
            trainable = objective_with_gpu, 
            param_space=search_space,
            resume_errored=True
        )
    else:
        tuner = tune.Tuner(  
            trainable = objective_with_gpu,
            tune_config=tune.TuneConfig(
                metric="inference_performance", mode="min", search_alg=algo,
                num_samples=cfg.hyperparameter_optimization.runs
            ),
            run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
            param_space=search_space,
        )
    
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == "__main__":
    main()