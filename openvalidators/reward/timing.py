
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
from typing import List
from .config import RewardModelType
from .reward import BaseRewardModel
from transformers import  AutoTokenizer, AutoModel

class TimeAwareRewardModel(BaseRewardModel):
    """
    A model that calculates rewards based on the input prompt, completion, and time.
    """
    @property
    def name(self) -> str: 
        """The name of the reward model."""
        return RewardModelType.time_aware.value

    def __init__(self, device: str):
        """
        The constructor for TimeAwareRewardModel class.

        Parameters:
           device (str): The device to which the model will be sent.
        """
        super().__init__()
        self.device = device

    def get_rewards(self, prompt: str, completions: List[str], name: str, timing: List[int]) -> torch.FloatTensor:
        """
        Calculate the rewards for multiple completions.

        Parameters:
           prompt (str): The prompt.
           completions (List[str]): The list of completions.
           name (str): The name.
           time (str): The time.

        Returns:
           torch.FloatTensor: The calculated rewards.
        """
        # Return the calculated rewards for all completions
        return torch.tensor([-time for time in timing], dtype=torch.float32).to(self.device)
    
        
    def apply(self, prompt: str, responses: List[bt.DendriteCall], name: str, test=False) -> torch.FloatTensor:
        """
        Applies the reward model across each call. Unsuccessful responses are assigned a reward of zero.

        Parameters:
            prompt (str): The prompt.
            responses (List[bt.DendriteCall]): The list of dendrite calls.
            name (str): The name.
            test (bool): A boolean indicating whether or not this is a test run. Default is False.

        Returns:
            torch.FloatTensor: The calculated rewards.
        """
        # Get indices of successful responses
        successful_completions_indices: List[int] = [idx for idx, resp in enumerate(responses) if resp.is_success]

        # Extract the completion text from successful responses
        successful_completions: List[str] = [responses[idx].completion.strip() for idx in successful_completions_indices]
        successful_timings: List[str] = [responses[idx].elapsed_time for idx in successful_completions_indices]

        # Calculate rewards for each successful completion
        successful_rewards = self.get_rewards(prompt, successful_completions, name, successful_timings)

        # Apply softmax normalization to the rewards
        successful_rewards = self.normalize_rewards(successful_rewards, test)

        # Initialize a tensor of zeros to hold the rewards for all responses
        filled_rewards = torch.zeros(len(responses), dtype=torch.float32)

        # Fill in the rewards for successful responses
        for idx, reward in zip(successful_completions_indices, successful_rewards):
            filled_rewards[idx] = reward

        # Return the tensor of rewards
        return filled_rewards
