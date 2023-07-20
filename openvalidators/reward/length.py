
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

class LengthAwareRewardModel(BaseRewardModel):
    """
    A model that calculates rewards based on the input prompt, completion, and the length of the completion.

    Attributes:
        reward_model_path (str): The path to the pretrained reward model.
        revision (str): The revision version of the pretrained model.
    """
    
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
        self.tokenizer = AutoTokenizer.from_pretrained(LengthAwareRewardModel.reward_model_path, 
                                                       revision = LengthAwareRewardModel.revision)

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
                                    truncation=True,
                                    ).to(self.device)
            # Multiply the reward by the length of the completion
            return len(inputs)

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

    