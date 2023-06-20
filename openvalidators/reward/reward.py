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

import torch
import bittensor as bt
from typing import List
from abc import abstractmethod

class BaseRewardModel:

    @property
    @abstractmethod
    def name(self) -> str: ...
    def __str__(self) -> str: return str(self.name)
    def __repr__(self) -> str: return str(self.name)
    def __init__( self ):
        self.mean = 0
        self.count = 0
        self.variance = 0
                 
    @abstractmethod
    def reward( self, prompt: str, completion: str ) -> float: ...

    def apply( self, prompt: str, responses: List[ bt.DendriteCall ]) -> torch.FloatTensor:
        """ Applies the reward model across each call. Non successful responses are zeroed.
        """

        # Get indices of correctly responding calls.
        successful_completions_indices: List[int] = [ idx for idx, resp in enumerate(responses) if resp.is_success ]

        # Get all completions from responding calls.
        successful_completions: List[str] = [ responses[idx].completion.strip() for idx in successful_completions_indices]

        # Reward each completion.
        successful_rewards = torch.tensor( [ self.reward( prompt, completion ) for completion in successful_completions ], dtype = torch.float32 )

        # Softmax rewards across samples.
        successful_rewards = successful_rewards.softmax(0)

        # Init zero rewards for all calls.
        filled_rewards = torch.zeros( len( responses ), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward in zip(successful_completions_indices, successful_rewards):
            filled_rewards[idx] = reward

        # Return the filled rewards.
        return filled_rewards 

    def update(self, sample: torch.FloatTensor):
        sample_mean = sample.mean()
        sample_variance = sample.var()

        w_distribution = self.count/ (self.count + len(sample))
        w_sample = len(sample) / (self.count+len(sample))
        self.mean = w_distribution*self.mean + w_sample* sample_mean
        self.variance = (w_distribution)*(self.variance) + (w_sample)*(sample_variance) + (w_distribution*w_sample)*(self.mean-sample_mean)**2 

    def normalize(self, sample: torch.FloatTensor):
        if self.variance != 0:
            rewards = sample-self.mean/self.variance
        else:
            rewards = sample
        self.update(sample)
        return rewards


class MockRewardModel( BaseRewardModel ):

    @property
    def name(self) -> str: return self.mock_name

    def __init__(self, mock_name: str = 'MockReward'):
        super().__init__()
        self.mock_name = mock_name

    def reward( self, prompt: str, completion: str ) -> float:
        return 0.0


        