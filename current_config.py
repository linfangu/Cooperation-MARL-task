from MultiAgentSync_fullobs_samefield_randnose import MultiAgentSync_fullobs
from MultiAgentSync_fullobs_samefield_randnose import MultiAgentSing_fullobs
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTorchPolicy
from Customcallback import CustomCallbacks
from simple_rnn_v2_3_2 import AnotherTorchRNNModel
import torch


def current_coop_config(limit=2):
    env_config = {"height": 8, "width": 8, "sync_limit": limit, "randomize": True}
    env = MultiAgentSync_fullobs(env_config)
    # ModelCatalog.register_custom_model("simple_fc", CustomFCNet)
    ModelCatalog.register_custom_model("rnn2", AnotherTorchRNNModel)

    policies = {
        "policy1": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
        "policy2": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
    }

    # 2) Defines an agent->policy mapping function.
    def policy_mapping_fn(agent_id: str) -> str:
        # Make sure agent ID is valid.
        assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
        ### Modify Code here ####
        id = agent_id[-1]
        return f"policy{id}"

    config = {
        "env": MultiAgentSync_fullobs,  # "my_env" <- if we previously have registered the env with `tune.register_env("[name]", lambda config: [returns env object])`.
        "env_config": env_config,
        "num_workers": 0,
        "exploration_config": {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            # "type": ParameterNoiseCuriosity,
            # "noise_stddev": 0.0005,
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 64,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            # FIXME
            "feature_net_config": {
                "use_lstm": False,
            },
            # "inverse_net_hiddens": [256,256],  # Hidden layers of the "inverse" model.
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            # FIXME
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            # "forward_net_hiddens": [256,256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            },
        },
        # !PyTorch users!
        "framework": "torch",  # If users have chosen to install torch instead of tf.
        "create_env_on_driver": True,
    }

    max_seq_len = 10

    # 3) RNN config.
    config.update(
        {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "model": {
                "custom_model": "rnn2",
                "max_seq_len": max_seq_len,
                "custom_model_config": {
                    # "fc_size": 128,
                    "rnn_hidden_size": 256,
                    "l2_lambda": 0.1,
                    "l2_lambda_inp": 0,
                    "noise_std": 1,
                    "device": torch.device("cuda:0"),
                },
            },
            "num_workers": 0,
            "num_gpus": 1,
            "callbacks": CustomCallbacks,
        }
    )

    # config.update({"kl_coeff": 0.0,
    #               "kl_target":0.0}) # remove kl

    print()
    print(f"agent1 is now mapped to {policy_mapping_fn('agent1')}")
    print(f"agent2 is now mapped to {policy_mapping_fn('agent2')}")
    return config, env_config


def current_sing_config():
    env_config = {
        "height": 8,
        "width": 8,
    }
    env = MultiAgentSing_fullobs(env_config)

    # ModelCatalog.register_custom_model("simple_fc", CustomFCNet)
    ModelCatalog.register_custom_model("rnn2", AnotherTorchRNNModel)

    policies = {
        "policy1": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
        "policy2": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
    }

    # 2) Defines an agent->policy mapping function.
    def policy_mapping_fn(agent_id: str) -> str:
        # Make sure agent ID is valid.
        assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
        ### Modify Code here ####
        id = agent_id[-1]
        return f"policy{id}"

    config = {
        "env": MultiAgentSing_fullobs,  # "my_env" <- if we previously have registered the env with `tune.register_env("[name]", lambda config: [returns env object])`.
        "env_config": env_config,
        "num_workers": 0,
        "exploration_config": {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            # "type": ParameterNoiseCuriosity,
            # "noise_stddev": 0.0005,
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 64,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            # FIXME
            "feature_net_config": {
                "use_lstm": False,
            },
            # "inverse_net_hiddens": [256,256],  # Hidden layers of the "inverse" model.
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            # FIXME
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            # "forward_net_hiddens": [256,256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            },
        },
        # !PyTorch users!
        "framework": "torch",  # If users have chosen to install torch instead of tf.
        "create_env_on_driver": True,
    }

    max_seq_len = 10

    # 3) RNN config.
    config.update(
        {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "model": {
                "custom_model": "rnn2",
                "max_seq_len": max_seq_len,
                "custom_model_config": {
                    # "fc_size": 128,
                    "rnn_hidden_size": 256,
                    "l2_lambda": 0.1,
                    "l2_lambda_inp": 0,
                    "noise_std": 1,
                    "device": torch.device("cuda:0"),
                },
            },
            "num_workers": 0,
            "num_gpus": 0.2,
            "callbacks": CustomCallbacks,
        }
    )

    print()
    print(f"agent1 is now mapped to {policy_mapping_fn('agent1')}")
    print(f"agent2 is now mapped to {policy_mapping_fn('agent2')}")
    return config, env_config
