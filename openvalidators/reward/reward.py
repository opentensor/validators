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

    @abstractmethod
    def reward( self, prompt: str, completion: str, name: str ) -> float: ...

    def __init__(self) -> None:
        self.old_count = 0
        self.old_mean = 0.0
        self.old_var = 0.0

    def normalize_rewards( self, rewards: torch.FloatTensor ) -> torch.FloatTensor:
        """
            This method normalizes the given rewards by updating the moving mean and variance statistics. The rewards are first standardized, and then scaled to the 0-1 range using a cumulative distribution function (CDF) to ensure they're in a comparable range across different environments.

            Args:
            rewards (torch.FloatTensor): The reward values to be normalized.

            Returns:
            torch.FloatTensor: The normalized reward values.

            Note:
            - This function uses Welford's online algorithm to update the mean and variance.
            - It standardizes the reward values using the updated mean and variance.
            - It then scales the standardized values to the 0-1 range using the error function (erf) as a CDF.
        """
        # Calculate the mean, count, and standard deviation of the new rewards.
        new_mean = rewards.mean()
        new_count = rewards.count()
        new_var = rewards.std()

        # Compute the weights for the new and old rewards.
        new_weight = new_count / (self.old_count + new_count)
        old_weight = self.old_count / (self.old_count + new_count)

        # Update the old mean with the new mean and weights.
        self.old_mean = new_weight * new_count + old_weight * self.old_mean
        # Update the old variance with the new variance and weights, and adjusting for the difference in means.
        self.old_var = (new_weight * new_var) + (old_weight * self.old_var) + (new_weight * old_weight) * (new_mean - self.old_mean) ** 2
        # Update the old count with the new count, but don't exceed the limit.
        self.old_count = min(self.count_limit, self.old_count + new_count)

        # Standardize the rewards using the updated mean and variance.
        rewards = (rewards - self.old_mean) / torch.sqrt(self.old_var)
        # Scale the standardized rewards to the range [0, 1] using the error function as a cumulative distribution function (CDF).
        rewards = 0.5 * (1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0]))))

        return rewards

    def apply( self, prompt: str, responses: List[ bt.DendriteCall ], name: str) -> torch.FloatTensor:
        """ Applies the reward model across each call. Unsuccessful responses are zeroed.
        """
        # Get indices of correctly responding calls.
        successful_completions_indices: List[int] = [ idx for idx, resp in enumerate(responses) if resp.is_success ]

        # Get all completions from responding calls.
        successful_completions: List[str] = [ responses[idx].completion.strip() for idx in successful_completions_indices]

        # Reward each completion.
        successful_rewards = torch.tensor( [ self.reward( prompt, completion, name ) for completion in successful_completions ], dtype = torch.float32 )

        # Softmax rewards across samples.
        successful_rewards = self.normalize_rewards( successful_rewards )

        # Init zero rewards for all calls.
        filled_rewards = torch.zeros( len( responses ), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward in zip(successful_completions_indices, successful_rewards):
            filled_rewards[idx] = reward

        # Return the filled rewards.
        return filled_rewards 


class MockRewardModel( BaseRewardModel ):

    @property
    def name(self) -> str: return self.mock_name

    def __init__(self, mock_name: str = 'MockReward'):
        super().__init__()
        self.mock_name = mock_name

    def reward( self, prompt: str, completion: str, name: str ) -> float:
        return 0.0


        