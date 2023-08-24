# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import argparse
import bittensor as bt
from loguru import logger
from openvalidators.gating import BaseGatingModel
from openvalidators.reward import DefaultRewardFrameworkConfig


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)
    bt.wallet.check_config(config)
    bt.subtensor.check_config(config)

    if config.mock:
        config.neuron.mock_reward_models = True
        config.neuron.mock_dendrite_pool = True
        config.neuron.mock_gating_model = True
        config.neuron.mock_dataset = True
        config.wallet._mock = True

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            config.neuron.full_path + "/" + "completions.log",
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    # Netuid Arg
    parser.add_argument("--netuid", type=int, help="Prompting network netuid", default=1)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name. ",
        default="core_prompting_validator",
    )
    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run the validator on.",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--neuron.disable_log_rewards",
        action="store_true",
        help="Disable all reward logging, suppresses reward functions and their values from being logged to wandb.",
        default=False,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.05,
    )

    parser.add_argument(
        "--neuron.followup_timeout",
        type=float,
        help="Follow up query timeout.",
        default=10,
    )
    parser.add_argument(
        "--neuron.followup_sample_size",
        type=int,
        help="How many miners to query for the follow up prompt.",
        default=50,
    )

    parser.add_argument("--neuron.answer_timeout", type=float, help="Answer query timeout.", default=10)
    parser.add_argument(
        "--neuron.answer_sample_size",
        type=int,
        help="How many miners to query for the answer.",
        default=50,
    )
    parser.add_argument(
        "--neuron.num_followup_steps",
        type=int,
        help="How many followup steps to take.",
        default=4,
    )

    parser.add_argument(
        "--neuron.epoch_length_override",
        type=int,
        help="Override the default epoch length (how often we set weights).",
        default=0,
    )
    parser.add_argument(
        "--neuron.checkpoint_block_length",
        type=int,
        help="Blocks before a checkpoint is saved.",
        default=10,
    )
    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )
    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument("--wandb.off", action="store_true", help="Turn off wandb.", default=False)
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="openvalidators",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="An entity is a username or team name where youre sending runs.",
        default="opentensor-dev",
    )
    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )
    parser.add_argument(
        "--wandb.weights_step_length",
        type=int,
        help="How many steps before we log the weights.",
        default=10,
    )
    parser.add_argument(
        "--wandb.run_step_length",
        type=int,
        help="How many steps before we rollover to a new run.",
        default=1500,
    )
    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )
    parser.add_argument(
        "--wandb.track_gating_model",
        action="store_true",
        help="Track the model weights of the gating model in wandb.",
        default=False,
    )

    # Mocks
    parser.add_argument("--mock", action="store_true", help="Mock all items.", default=False)
    parser.add_argument(
        "--neuron.mock_reward_models",
        action="store_true",
        help="Dont download the reward model.",
        default=False,
    )
    parser.add_argument(
        "--neuron.blacklist_off",
        action="store_true",
        help="Dont apply the blacklist reward model",
        default=False,
    )
    parser.add_argument(
        "--neuron.nsfw_off",
        action="store_true",
        help="Dont apply the nsfw reward model",
        default=False,
    )
    parser.add_argument(
        "--neuron.relevance_off",
        action="store_true",
        help="Dont apply the relevance reward model",
        default=False,
    )
    parser.add_argument(
        "--neuron.diversity_off",
        action="store_true",
        help="Dont apply the diversity reward model",
        default=False,
    )
    parser.add_argument(
        "--neuron.task_validator_off",
        action="store_true",
        help="Dont apply the task validator reward model",
        default=False,
    )    

    parser.add_argument(
        "--reward.reciprocate_weight",
        type=float,
        help="Weight for the reciprocate reward model",
        default=DefaultRewardFrameworkConfig.reciprocate_model_weight,
    )
    parser.add_argument(
        "--reward.dpo_weight",
        type=float,
        help="Weight for the dpo reward model",
        default=DefaultRewardFrameworkConfig.dpo_model_weight,
    )
    parser.add_argument(
        "--reward.rlhf_weight",
        type=float,
        help="Weight for the rlhf reward model",
        default=DefaultRewardFrameworkConfig.rlhf_model_weight,
    )
    parser.add_argument(
        "--reward.dahoas_weight",
        type=float,
        help="Weight for the dahoas reward model",
        default=DefaultRewardFrameworkConfig.dahoas_model_weight,
    )
    parser.add_argument(
        "--reward.prompt_based_weight",
        type=float,
        help="Weight for the prompt-based reward model",
        default=DefaultRewardFrameworkConfig.prompt_model_weight,
    )

    parser.add_argument(
        "--neuron.mock_dendrite_pool",
        action="store_true",
        help="Dont download the dendrite pool.",
        default=False,
    )
    parser.add_argument(
        "--neuron.mock_gating_model",
        action="store_true",
        help="Dont download the gating model.",
        default=False,
    )
    parser.add_argument(
        "--neuron.mock_dataset",
        action="store_true",
        help="Dont download the dataset.",
        default=False,
    )
    parser.add_argument(
        "--neuron.use_custom_gating_model",
        action="store_true",
        help="Use a custom gating model.",
        default=False,
    )


def config(cls):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    BaseGatingModel.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
