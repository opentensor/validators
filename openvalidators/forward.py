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
import torch
import random
import bittensor as bt
import random

from loguru import logger
from typing import List
from dataclasses import asdict
from openvalidators.event import EventSchema
from openvalidators.misc import ttl_get_block
from openvalidators.prompts import followup_request_template, augment_request_template, school_levels
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

async def run_step( self, prompt: str, k: int, timeout: float, name: str, exclude: list = []):
    bt.logging.debug( "run_step", name )

    # Record event start time.
    event = {'name': name}
    start_time = time.time()

    # Get the list of uids to query for this step.
    uids = get_random_uids( self, k = k , exclude=exclude).to(self.device)

    # Make calls to the network with the prompt.
    responses: List[ bt.DendriteCall ] = await self.dendrite_pool.async_forward( 
        uids = uids,
        roles = [ "user" ],
        messages = [ prompt ],
        timeout = timeout,
    )

    # Compute the rewards for the responses given the prompt.
    rewards:torch.FloatTensor = torch.ones( len( responses ), dtype=torch.float32).to(self.device) 
    for reward_fn_i in self.reward_functions:
        reward_i = reward_fn_i.apply( prompt, responses, name ).to( self.device )
        rewards += reward_i
        if self.config.neuron.log_rewards:     
            event[ reward_fn_i.name ] = reward_i.tolist()
        bt.logging.trace( str(reward_fn_i.name), reward_i.tolist() )
    
    for masking_fn_i in self.masking_functions:
        mask_i = masking_fn_i.apply( prompt, responses, name ).to( self.device )
        rewards *= mask_i
        if self.config.neuron.log_rewards:   
            event[ masking_fn_i.name ] = mask_i.tolist()
        bt.logging.trace( str(masking_fn_i.name), mask_i.tolist() )

    # Train the gating model based on the predicted scores and the actual rewards.
    gating_scores: torch.FloatTensor = self.gating_model( prompt ).to(self.device)
    gating_loss: torch.FloatTensor = self.gating_model.backward( scores = gating_scores[uids], rewards = rewards )

    # Find the best completion given the rewards vector.
    completions: List[str] = [comp.completion for comp in responses ]
    best:str = completions[ rewards.argmax( dim = 0 )].strip()

    print('prompt', prompt)
    for ind, comp in enumerate(completions):
        print('********')
        print('reward:',comp, rewards[ind])
        print('********')

    # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter( 0, uids, rewards ).to(self.device) 

    # Update moving_averaged_scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha:float = self.config.neuron.moving_average_alpha
    self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (1 - alpha) * self.moving_averaged_scores.to(self.device)

    # Log the step event.
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
    if not self.config.neuron.dont_save_events: logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    wandb_event = EventSchema.from_dict(event, self.config.neuron.log_rewards)
    if not self.config.wandb.off: self.wandb.log(asdict(wandb_event))

    # Return the event.
    return event

    
async def forward(self):

    # Obtain a unique context from the dataset.
    data = next(self.dataset)["text"]

    # Truncate context to a limited set of sentences.
    bootstrap_prompt = '.'.join(data.split('.', maxsplit=20)[:-1])

    # Form the augment prompt, requesting a summary at a random school level.
    random_level = random.randint(0, 4)
    augment_prompt = f"{bootstrap_prompt}\n\n{augment_request_template} {school_levels[random_level]} level.\n\n"

    # Request a summary, given the original context.
    augment_event = await run_step( 
        self, 
        prompt = augment_prompt, 
        name = 'augment',
        k = self.config.neuron.followup_sample_size,
        timeout = self.config.neuron.followup_timeout,
    )

    # Get a followup question, given the summarized context.
    prompt = f"{augment_event['best']}\n\n{followup_request_template}\n\n"
    followup_event = await run_step( 
        self, 
        prompt = prompt, 
        name = 'followup',
        k = self.config.neuron.followup_sample_size,
        timeout = self.config.neuron.followup_timeout,
        exclude = augment_event['uids']
    )

    # Ask the followup question, given the original context.
    prompt = f"{bootstrap_prompt}\n\n{followup_event['best']}"
    answer_event = await run_step( 
        self, 
        prompt = prompt, 
        name = 'answer',
        k = self.config.neuron.answer_sample_size,
        timeout = self.config.neuron.answer_timeout,
        exclude = augment_event['uids'] + followup_event['uids']
    )
