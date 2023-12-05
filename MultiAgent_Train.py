# Import libraries
import argparse
import datetime
import os
import sys
import importlib
import numpy as np
import pickle
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms import ppo
from ray.rllib.policy import policy
from ray.tune import CLIReporter
from ray import air
from ray import train
from ray import tune
import pprint
from current_config import current_coop_config, current_sing_config
from MultiAgentSync_fullobs_samefield_randnose import (
    MultiAgentSync_fullobs,
    MultiAgentSing_fullobs,
)

# provide arguments through command line
parser = argparse.ArgumentParser(
    description="Train RL agent for cooperation with composite curriculum"
)
parser.add_argument(
    "--dir-out",
    default="/home/kumquat/Documents/Linfan/MARL/trainings/",
    type=str,
    help="Full path to output directory",
)
parser.add_argument(
    "--name",
    default="mytrainings",
    type=str,
    help="Name of traning",
)
parser.add_argument(
    "--stage",
    type=str,
    default="Single",  # Single, Coop1, Coop2
    help="Which environment to train",
)

parser.add_argument("--num-gpus", type=float, default=1, help="number of GPUs deployed")
parser.add_argument(
    "--train-iter", type=int, default=6000, help="number of training iterations"
)
parser.add_argument(
    "--checkpoint-freq", type=int, default=10, help="freq of checkpoints"
)
parser.add_argument(
    "--CUDA-VISIBLE-DEVICES", type=str, default="0", help="Specify the gpu to use"
)

parser.add_argument("--resume", type=bool, default=False, help="if resume from a path")
parser.add_argument("--resume-path", type=str, default=None, help="experiment path")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args["CUDA_VISIBLE_DEVICES"]
    if args["stage"] == "Single":
        config, env_config = current_sing_config()

    elif args["stage"] == "Coop1":
        config, env_config = current_coop_config(limit =5)

    elif args["stage"] == "Coop2":
        config, env_config = current_coop_config()
        env = MultiAgentSing_fullobs(env_config)

    ray.init(num_gpus=1)

    progress_reporter = CLIReporter(
        max_progress_rows=10, max_report_frequency=300, infer_limit=8
    )
    progress_reporter.add_metric_column("policy_reward_mean")

    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=5=arg["checkpoint-freq"],
        # num_to_keep=100,
        checkpoint_at_end=True,
    )

    stop = {
        # Note that the keys used here can be anything present in the above `rllib_trainer.train()` output dict.
        "training_iteration": args["train_iter"],
    }

    print(
        f"Training with tune for {args['train_iter']} iterations, saving checkpoint every {args['checkpoint_freq']}"
    )
    if ray.__version__ == "2.2.0":
        run_config = air.RunConfig(
            name=args["name"],
            stop=stop,
            verbose=1,
            checkpoint_config=checkpoint_config,
            progress_reporter=progress_reporter,
            local_dir=args["dir_out"],
        )
    else:
        run_config = air.RunConfig(
            name=args["name"],
            stop=stop,
            verbose=1,
            checkpoint_config=checkpoint_config,
            progress_reporter=progress_reporter,
            storage_path=args["dir_out"],
        )

    agent_algorithm = "PPO"
  
    pprint.pprint(config)
    if not args["resume"]:
        tuner = tune.Tuner(
            agent_algorithm,
            param_space=config,
            run_config=run_config,
        )
    else:
        if ray.__version__ == "2.2.0":
            tuner = tune.Tuner(
                agent_algorithm,
                param_space=config,
            ).restore(
                args["resume_path"],
                resume_unfinished=True,
                resume_errored=True,
            )
        if ray.__version__ >= "2.6.0":
            tuner = tune.Tuner.restore(
                args["resume_path"],
                trainable=agent_algorithm,
                resume_unfinished=True,
                resume_errored=True,
                param_space=config,
            )

    results = tuner.fit()
    print(results)
    assert results.num_errors == 0


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
