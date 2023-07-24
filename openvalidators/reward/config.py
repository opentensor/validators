# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
from dataclasses import dataclass

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

from enum import Enum


class RewardModelType(Enum):
    rlhf = 'rlhf_reward_model'
    reciprocate = 'reciprocate_reward_model'
    dahoas = 'dahoas_reward_model'
    diversity = 'diversity_reward_model'
    prompt = 'prompt_reward_model'
    blacklist = 'blacklist_filter'
    nsfw = 'nsfw_filter'
    relevance = 'relevance_filter'
    time_aware = 'time_aware_reward_model'
    length_aware = 'length_aware_reward_model'

@dataclass(frozen=True)
class DefaultRewardFrameworkConfig:
    """Reward framework default configuration.
    Note: All the weights should add up to 1.0.
    """
    rlhf_model_weight: float = 0.6
    reciprocate_model_weight: float = 0.4
    dahoas_model_weight: float = 0
    prompt_model_weight: float = 0
