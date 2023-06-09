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
import copy
import pytest
import sys
from openvalidators.neuron import neuron as Neuron
from openvalidators.forward import forward

CLI_ARGS = 'validators/openvalidators/neuron.py --mock --wallet.mock --wandb.off'.split(' ')

def test_uid_weights_unchanged_unless_queried():
    
    sys.argv = CLI_ARGS
    neuron = Neuron()
    for _ in range(100):
        prev_scores = copy.deepcopy(neuron.moving_averaged_scores)
        # run forward pass
        event = forward()
        next_scores = neuron.moving_averaged_scores
        ignored_uids = [uid for uid in torch.arange(neuron.metagraph.n.item()) if uid not in event['followup_uids']+event['answer_uids']]
        
        assert next_scores[ignored_uids] == prev_scores[ignored_uids], "Unqueried uids should not change"

def test_weight_decay():
    # make a neuron with mock everything (reward, )
    neuron = neuron()
    for _ in range(100):
        prev_scores = copy.deepcopy(neuron.moving_averaged_scores)
        # run forward pass
        event = forward()
        next_scores = neuron.moving_averaged_scores
        ignored_uids = [uid for uid in range(neuron.metagraph.n.item()) if uid not in event['followup_uids']+event['answer_uids']]
        
        assert next_scores[ignored_uids] == prev_scores[ignored_uids], "Unqueried uids should not change"
        assert next_scores[event['followup_uids']] < prev_scores[event['followup_uids']], "Followup uids should decay"
        assert next_scores[event['answer_uids']] > prev_scores[event['answer_uids']], "Answer uids should increase"