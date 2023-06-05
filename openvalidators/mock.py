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
import asyncio
import bittensor as bt
from openvalidators.prompts import (
    scoring_mock_response,
    firewall_mock_response,
    is_firewall,
    is_scoring,
)
from openvalidators.gating import BaseGatingModel
from typing import List
from collections.abc import Iterator


class MockGatingModel(BaseGatingModel):
    def __init__(self, num_uids: int):
        super(MockGatingModel, self).__init__()
        # super(MockGatingModel, self).__init__()
        self.num_uids = num_uids
        self.linear = torch.nn.Linear(256, 10)

    def forward(self, message: str) -> "torch.FloatTensor":
        return torch.randn(self.num_uids)

    def backward(self, scores: torch.FloatTensor, rewards: torch.FloatTensor):
        return torch.tensor(0.0)

    def resync(
        self,
        previous_metagraph: "bt.metagraph.Metagraph",
        metagraph: "bt.metagraph.Metagraph",
    ):
        pass


class MockRewardModel(torch.nn.Module):
    def reward(
        self,
        completions_with_prompt: List[str],
        completions_without_prompt: List[str],
        difference=False,
        shift=3,
    ) -> torch.FloatTensor:
        return torch.randn(len(completions_with_prompt))


class MockDataset(Iterator):
    def __next__(self):
        return {"context": "What is the capital of Texas?"}


class MockDendriteResponse:
    completion = ""
    elapsed_time = 0

    def __init__(self, message: str):
        if is_firewall(message):
            self.completion = firewall_mock_response()
        elif is_scoring(message):
            self.completion = scoring_mock_response()
        else:
            self.completion = "The capital of Texas is Austin."

    def __str__(self):
        return f"MockDendriteResponse({self.completion})"

    def __repr__(self):
        return f"MockDendriteResponse({self.completion})"


class MockDendritePool(torch.nn.Module):
    def forward(self, roles: List[str], messages: List[str], uids: List[int], timeout: float):
        return [MockDendriteResponse(messages[0]) for _ in uids]

    async def async_forward(
        self,
        roles: List[str],
        messages: List[str],
        uids: List[int],
        timeout: float = 12,
        return_call=True,
    ):
        async def query():
            await asyncio.sleep(0.01)
            return [MockDendriteResponse(messages[0]) for _ in uids]

        return await query()

    def resync(self, metagraph):
        pass
