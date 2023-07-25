
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
import bittensor as bt

class LengthAwareRewardModel(BaseRewardModel):
    """
    A model that calculates rewards based on the input prompt, completion, and the length of the completion.

    Attributes:
        reward_model_path (str): The path to the pretrained reward model.
        revision (str): The revision version of the pretrained model.
    """
    reward_model_path: str = "EleutherAI/gpt-j-6b"

    @property
    def name(self) -> str: 
        """The name of the reward model."""
        return RewardModelType.length_aware.value

    def __init__(self, device: str):
        """
        The constructor for LengthAwareRewardModel class.

        Parameters:
           device (str): The device to which the model will be sent.
        """
        super().__init__()
        self.device = device
        # Load the tokenizer from the pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(LengthAwareRewardModel.reward_model_path)
        self.count = {}
        self.mean = {}
        self.var = {}

    def reward(self, prompt: str, completion: str, name: str) -> float:
        """
        Calculate the reward for a single prompt and completion. The length of the completion is used as an additional factor.

        Parameters:
           prompt (str): The prompt.
           completion (str): The completion.
           name (str): The name.

        Returns:
           float: The calculated reward.
        """
        with torch.no_grad():
            # Tokenize the message
            inputs = self.tokenizer(completion,
                                    return_tensors="pt",
                                    ).to(self.device)
            # Multiply the reward by the length of the completion
            token_len = inputs['input_ids'].size()[1]
            if name not in self.mean:
                cutoff = 300
            else:
                cutoff = self.mean[name] + 2*torch.sqrt(self.var[name])

            if token_len > cutoff:
                return torch.tensor([cutoff]).to(self.device)
            else:
                return token_len

    def get_rewards(self, prompt: str, completions: List[str], name: str) -> torch.FloatTensor:
        """
        Calculate the rewards for multiple completions. The length of each completion is used as an additional factor.

        Parameters:
           prompt (str): The prompt.
           completions (List[str]): The list of completions.
           name (str): The name.

        Returns:
           torch.FloatTensor: The calculated rewards.
        """
        # Return the calculated rewards for all completions
        return torch.tensor([self.reward(prompt, completion, name) for completion in completions], dtype=torch.float32).to(self.device)

    def apply(self, prompt: str, responses: List[bt.DendriteCall], name: str) -> torch.FloatTensor:
        """
        Applies the reward model across each call. Unsuccessful responses are assigned a reward of zero.

        Parameters:
            prompt (str): The prompt.
            responses (List[bt.DendriteCall]): The list of dendrite calls.
            name (str): The name.

        Returns:
            torch.FloatTensor: The calculated rewards.
        """
        # Get indices of successful responses
        successful_completions_indices: List[int] = [idx for idx, resp in enumerate(responses) if resp.is_success]

        # Extract the completion text from successful responses
        successful_completions: List[str] = [responses[idx].completion.strip() for idx in successful_completions_indices]

        # Calculate rewards for each successful completion
        successful_rewards = self.get_rewards(prompt, successful_completions, name)

        # Apply normalization to the rewards
        successful_rewards = self.normalize_rewards(successful_rewards, name)

        # Initialize a tensor of zeros to hold the rewards for all responses
        filled_rewards = torch.zeros(len(responses), dtype=torch.float32)

        # Fill in the rewards for successful responses
        for idx, reward in zip(successful_completions_indices, successful_rewards):
            filled_rewards[idx] = reward

        # Return the tensor of rewards
        return filled_rewards

    def normalize_rewards(self, rewards: torch.FloatTensor, name: str) -> torch.FloatTensor:
        """
        This method normalizes the given rewards by updating the moving mean and
        variance statistics. The rewards are first standardized, and then scaled to
        the 0-1 range using a cumulative distribution function (CDF) to ensure they're
        in a comparable range across different environments.

        Args:
            rewards (torch.FloatTensor): The reward values to be normalized.
            name (str): The name to be used as a key for indexing the mean, variance, and count.

        Returns:
            torch.FloatTensor: The normalized reward values.

        Note:
            - This function uses Welford's online algorithm to update the mean and variance.
            - It standardizes the reward values using the updated mean and variance.
            - It then scales the standardized values to the 0-1 range using the error
            function (erf) as a CDF.
        """        
        # Initialize the mean, variance, and count for a new name
        if name not in self.mean:
            self.mean[name] = 0.0
            self.var[name] = 0.0
            self.count[name] = 0

        # Get the number of rewards (successful responses).
        new_count = rewards.numel()

        # Update stats only if there are new rewards.
        if 0 < new_count and 0 < self.count[name] + new_count:
            # Calculate the mean and standard deviation of the new rewards.
            new_mean = rewards.mean()
            new_var = rewards.var(dim=0)

            # Compute the weights for the new and old rewards.
            new_weight = new_count / (self.count[name] + new_count)
            old_weight = self.count[name] / (self.count[name] + new_count)

            # Save the difference in means before updating the old mean.
            diff = new_mean - self.mean[name]

            # Update the old mean with the new mean and weights.
            self.mean[name] = new_weight * new_mean + old_weight * self.mean[name]
            # Update the old variance with the new variance and weights, and adjusting for the difference in means.
            self.var[name] = (
                new_weight * new_var
                + old_weight * self.var[name]
                + new_weight * old_weight * diff * diff
            )
            # Update the old count with the new count, but don't exceed the limit.
            self.count[name] = min(self.count_limit, self.count[name] + new_count)

        # Standardize the rewards using the updated mean and variance.
        rewards = rewards - self.mean[name]
        if self.var[name] > 0:
            rewards /= torch.sqrt(self.var[name])
        # Scale the standardized rewards to the range [0, 1] using the error function as a cumulative distribution function (CDF).
        rewards = 0.5 * (1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device)))

        return rewards

    