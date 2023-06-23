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

import sys
import copy
import torch
import asyncio
import bittensor as bt
from datasets import load_dataset
import random

from openvalidators.dendrite import AsyncDendritePool
from openvalidators.gating import GatingModel, SentenceEmbedGatingModel
from openvalidators.mock import MockDendritePool, MockDataset, MockRewardModel, MockGatingModel

# Load local forward function.
from openvalidators.config import add_args, check_config, config
from openvalidators.run import run
from openvalidators.misc import ttl_get_block
from openvalidators.utils import init_wandb

# Load gating models
from openvalidators.reward import (
    Blacklist, 
    NSFWRewardModel, 
    OpenAssistantRewardModel, 
    ReciprocateRewardModel, 
    BertRelevanceRewardModel, 
    MockRewardModel, 
    DahoasRewardModel,
    DiversityRewardModel,
    PromptRewardModel
)

class neuron:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def run(self):
        run(self)

    def __init__(self):
        self.config = neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        print(self.config)
        bt.logging.info("neuron.__init__()")

        # Init device.
        bt.logging.debug("loading", "device")
        self.device = torch.device(self.config.neuron.device)
        bt.logging.debug(str(self.device))

        # Init subtensor
        bt.logging.debug("loading", "subtensor")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        # Init wallet.
        bt.logging.debug("loading", "wallet")
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()
        if not self.config.wallet._mock:
            self.wallet.reregister(subtensor=self.subtensor, netuid=self.config.netuid)
        bt.logging.debug(str(self.wallet))

        # Init metagraph.
        bt.logging.debug("loading", "metagraph")
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        bt.logging.debug(str(self.metagraph))

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(str(self.moving_averaged_scores))

        # Dataset: used to generate the base prompts ( initial randomness. )
        bt.logging.debug("loading", "dataset")
        if self.config.neuron.mock_dataset:
            self.dataset = MockDataset()
        else:
            seed = random.randint(0,1000)
            self.dataset = iter(load_dataset("openwebtext", split="train", streaming=True).shuffle(seed=seed, buffer_size=100000))
        bt.logging.debug(str(self.dataset))

        # Init the gating model which learns which miners to select for each query.
        bt.logging.debug("loading", "gating_model")
        if self.config.neuron.mock_gating_model:
            self.gating_model = MockGatingModel(self.metagraph.n.item())
        elif self.config.neuron.use_custom_gating_model:
            self.gating_model = SentenceEmbedGatingModel(metagraph=self.metagraph, config=self.config).to(self.device)
        else:
            self.gating_model = GatingModel(metagraph=self.metagraph, config=self.config).to(self.device)
        bt.logging.debug(str(self.gating_model))

        # Dendrite pool for querying the network during training.
        bt.logging.debug("loading", "dendrite_pool")
        if self.config.neuron.mock_dendrite_pool:
            self.dendrite_pool = MockDendritePool()
        else:
            self.dendrite_pool = AsyncDendritePool(keypair=self.wallet.hotkey, metagraph=self.metagraph)
        bt.logging.debug(str(self.dendrite_pool))

        # Init Reward model
        bt.logging.debug("loading", "reward_functions")
        if self.config.neuron.mock_reward_models:
            self.reward_functions = []
            self.masking_functions = [
                MockRewardModel('blacklist_filter'), 
                MockRewardModel('nsfw_filter') 
            ]
            bt.logging.debug(str(self.reward_functions))
        else:
            self.reward_weights = torch.tensor([self.config.neuron.openassistant, self.config.neuron.reciprocate,
                                                self.config.neuron.dahoas, self.config.neuron.diversity,
                                                self.config.neuron.prompt_based], dtype=torch.float32).to(self.device)
            # Ensure reward function weights sum to 1.
            if self.reward_weights.sum() != 1:
                sys.exit('Reward function weights do not sum to 1.')

            self.reward_functions = [ 
                OpenAssistantRewardModel( device = self.device ) if self.config.neuron.openassistant > 0 else MockRewardModel('rlhf_reward_model'),
                ReciprocateRewardModel( device = self.device ) if self.config.neuron.reciprocate > 0 else MockRewardModel('reciprocate_reward_model'),
                DahoasRewardModel( path = self.config.neuron.full_path, device = self.device ) if self.config.neuron.dahoas > 0 else MockRewardModel('dahoas_reward_model'),
                DiversityRewardModel( device = self.device ) if self.config.neuron.diversity > 0 else MockRewardModel('diversity_reward_model'),
                PromptRewardModel( device = self.device ) if self.config.neuron.prompt_based > 0 else MockRewardModel('prompt_reward_model'),
            ]
            self.masking_functions = [
                Blacklist() if not self.config.neuron.blacklist_off else MockRewardModel('blacklist'), 
                BertRelevanceRewardModel( device = self.device ) if not self.config.neuron.relevance_off else MockRewardModel('relevance'),
                NSFWRewardModel( device = self.device ) if not self.config.neuron.nsfw_off else MockRewardModel('nsfw'),
            ]
            bt.logging.debug(str(self.reward_functions))

        # Init the event loop.
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        if not self.config.wandb.off:
            bt.logging.debug("loading", "wandb")
            init_wandb(self)

        if self.config.neuron.epoch_length_override:
            self.config.neuron.epoch_length = self.config.neuron.epoch_length_override
        else:
            self.config.neuron.epoch_length = self.subtensor.validator_epoch_length(self.config.netuid)

        self.prev_block = ttl_get_block(self)
        self.step = 0


def main():
    neuron().run()


if __name__ == "__main__":
    main()
