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

import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator

def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Dataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0,1000)
        self.openwebtext = iter( load_dataset("openwebtext", split="train", streaming=True).shuffle(seed=seed, buffer_size=100000) )
        self.red_pajama = iter( load_dataset("togethercomputer/RedPajama-Data-1T", split='Train', streaming=True).shuffle(seed=seed, buffer_size=100000) )

    def __next__(self):
        if random.random() < 0.5:
            return {"text": next(self.openwebtext)["text"]}
        elif random.random() > 0.5:
            # Red Pajama is a bit too long, so we split it into chunks of 300 words.
            next_text = next(self.red_pajama)["text"]
            next_split = next_text.split(" ")
            chunks = list(chunk(next_split, 300))
            return {"text": random.choice(chunks)}


class MockDataset(Iterator):
    def __next__(self):
        return {"text": "What is the capital of Texas?"}


   