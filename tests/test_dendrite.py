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
import copy
import unittest
from unittest.mock import MagicMock
from openvalidators.dendrite import AsyncDendritePool


class DendriteTestCase(unittest.TestCase):
    def setUp(self):
        """
        Creates a mock metagraph with 1024 mock axons.
        """
        mock_metagraph = MagicMock(spec=bt.metagraph)
        mock_metagraph.uids = torch.tensor(range(0, 1024))
        mock_metagraph.axons = [
            bt.axon_info(
                version=0,
                ip="0.0.0.0/0",
                port=12345,
                ip_type=0,
                hotkey=str(num),
                coldkey=str(num)
            ) for num in range(0, 1024)
        ]

        self.metagraph = mock_metagraph
        self.keypair = "test"

    def test_resync_uid_modified_metagraph(self):
        # Arrange: Creates Async dendrite pool and modifies the metagraph by changing the axon_info at defined index
        dendrite_pool = AsyncDendritePool(keypair=self.keypair, metagraph=self.metagraph)

        # Modify the hotkey of the first axon of the metagraph
        index = 0
        modified_metagraph = copy.deepcopy(self.metagraph)
        modified_metagraph.axons[index].hotkey = "hotkey-test"

        # Act: Resync the dendrite pool with the modified metagraph
        dendrite_pool.resync(modified_metagraph)

        # Assert: The dendrite pool hotkeys should be the same as the modified metagraph hotkeys after resync
        dendrite_hot_keys = list(map(lambda dendrite: dendrite.axon_info.hotkey, dendrite_pool.dendrites))
        new_metagraph_hot_keys = list(map(lambda axon: axon.hotkey, modified_metagraph.axons))

        self.assertEqual(dendrite_hot_keys, new_metagraph_hot_keys)

    def test_resync_uid_add(self):
        original_metagraph = self.metagraph

        smaller_metagraph = copy.deepcopy(original_metagraph)

        # Remove the last axon from the metagraph
        smaller_metagraph.axons.pop()

        # Creates dendrite pool with smaller metagraph
        dendrite_pool = AsyncDendritePool(keypair=self.keypair, metagraph=smaller_metagraph)

        # Resync the dendrite pool with the original metagraph, that has one more axon
        dendrite_pool.resync(original_metagraph)

        assert len(dendrite_pool.dendrites) == len(original_metagraph.axons)

        dendrite_hot_keys = list(map(lambda dendrite: dendrite.axon_info.hotkey, dendrite_pool.dendrites))
        new_metagraph_hot_keys = list(map(lambda axon: axon.hotkey, original_metagraph.axons))

        self.assertEqual(dendrite_hot_keys, new_metagraph_hot_keys)


if __name__ == "__main__":
    unittest.main()
