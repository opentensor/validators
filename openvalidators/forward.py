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
# DEALINGS IN
#  THE SOFTWARE.

import time 
import wandb
import torch
import random
import bittensor as bt

from loguru import logger
from typing import List, Union, Dict
from openvalidators.misc import ttl_get_block
from openvalidators.prompts import (
    extract_score,
    followup_request_template,
    answer_scoring_template,
    followup_scoring_template,
)
from openvalidators.utils import check_uid_availability

def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available and uid_is_not_excluded:
            candidate_uids.append(uid)

    available_uids = torch.tensor(candidate_uids, dtype=torch.int64).to(self.device)
    uids = torch.tensor(random.sample(available_uids.tolist(), k), dtype=torch.int64)
    return uids

async def run_step( self, prompt: str, k: int, timeout: float, name: str ):
    bt.logging.debug( "run_step", name )

    # Record event start time.
    event = {'name': name}
    start_time = time.time()


    # Get the list of uids to query for this step.
    uids = get_random_uids( self, k = k ).to(self.device)

    # Make calls to the network with the prompt.
    responses: List[ bt.DendriteCall ] = await self.dendrite_pool.async_forward( 
        uids = uids,
        roles = [ "user" ],
        messages = [ prompt ],
        timeout = timeout,
    )

    # Compute the rewards for the responses gien the prompt.
    rewards:torch.FloatTensor = torch.ones( len( responses ), dtype=torch.float32).to(self.device) 
    for reward_fn_i in self.reward_functions:
        rewards *= reward_fn_i.apply( prompt, responses ).to(self.device) 
        # NOTE(const) Turning this off to reduce the event size.
        # event[ reward_fn_i.name ] = rewards.tolist()

    # Train the gating model based on the predicted scores and the actual rewards.
    gating_scores: torch.FloatTensor = self.gating_model( prompt ).to(self.device)
    gating_loss: torch.FloatTensor = self.gating_model.backward( scores = gating_scores[uids], rewards = rewards )

    # Find the best completion given the rewards vector.
    completions: List[str] = [comp.completion for comp in responses ]
    best:str = completions[ rewards.argmax( dim = 0 )].strip()

    # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter( 0, uids, rewards ).to(self.device) 

    # Update moving_averaged_scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha:float = self.config.neuron.moving_average_alpha
    self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(self.device)

    # Log the step event..
    event.update({
        "block": ttl_get_block( self ),
        'step_length': time.time() - start_time,
        'prompt': prompt,
        'uids': uids.tolist(),
        'completions': completions,
        'rewards': rewards.tolist(),
        'gating_loss': gating_loss.item(),
        'best': best
    })
    bt.logging.debug( "event:", str(event) )
    if not self.config.wandb.off: self.wandb.log( event )
    if not self.config.neuron.dont_save_events: logger.log("EVENTS", "events", **event)

    # Return the event.
    return event

    
async def forward(self):

    # Query the network for the best follow up
    bootstrap_prompt = next(self.dataset)["context"]

    # Get a follow up.
    prompt = f"{bootstrap_prompt}\n\n{followup_request_template}\n\n"
    followup_event = await run_step( 
        self, 
        prompt = prompt, 
        name = 'followup',
        k = self.config.neuron.followup_sample_size,
        timeout = self.config.neuron.followup_timeout,
    )

    # Ask the follow up.
    prompt = f"{bootstrap_prompt}\n\n{followup_event['best']}"
    answer_event = await run_step( 
        self, 
        prompt = prompt, 
        name = 'answer',
        k = self.config.neuron.answer_sample_size,
        timeout = self.config.neuron.answer_timeout 
    )
