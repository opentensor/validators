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

# Utils for checkpointing and saving the model.
import torch
import wandb
import copy
import bittensor as bt
import openvalidators
from openvalidators.misc import ttl_get_block
from openvalidators.reward import MockRewardModel


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return not self.config.wandb.off and self.step and self.step % self.config.wandb.run_step_length == 0


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    tags = [self.wallet.hotkey.ss58_address, openvalidators.__version__, str(openvalidators.__spec_version__)]
    if self.config.mock:
        tags.append("mock")
    if self.config.neuron.use_custom_gating_model:
        tags.append("custom_gating_model")
    for fn in self.reward_functions:
        if not isinstance(fn, MockRewardModel):
            tags.append(str(fn.name))
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=self.config.wandb.project_name,
        entity=self.config.wandb.entity,
        config=self.config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def should_checkpoint(self):
    # Check if enough epoch blocks have elapsed since the last checkpoint.
    return (
        ttl_get_block(self) % self.config.neuron.checkpoint_block_length
        < self.prev_block % self.config.neuron.checkpoint_block_length
    )


def checkpoint(self):
    """Checkpoints the training process."""
    bt.logging.info("checkpoint()")
    resync_metagraph(self)
    save_state(self)


def resync_metagraph(self: 'openvalidators.neuron.neuron'):
    """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
    bt.logging.info("resync_metagraph()")

    # Copies state of metagraph before syncing.
    previous_metagraph = copy.deepcopy(self.metagraph)

    # Sync the metagraph.
    self.metagraph.sync(subtensor=self.subtensor)

    # Check if the metagraph axon info has changed.
    metagraph_axon_info_updated = previous_metagraph.axons != self.metagraph.axons

    if metagraph_axon_info_updated:
        bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        # Reconstruct the dendrite pool with the new endpoints.
        self.dendrite_pool.resync(self.metagraph)

        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.moving_averaged_scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            new_moving_average[: len(self.hotkeys)] = self.moving_averaged_scores
            self.moving_averaged_scores = new_moving_average

        # Resize the gating model.
        bt.logging.info("Re-syncing gating model")
        self.gating_model.resync(previous_metagraph, self.metagraph)

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)


def resync_linear_layer(
    linear_layer: torch.nn.Module,
    previous_metagraph: "bt.metagraph.Metagraph",
    metagraph: "bt.metagraph.Metagraph",
):
    """Resync the linear layer with the latest state of the network
    Args:
         linear_layer (:obj: torch.nn.Module): Linear layer to be resynced
         previous_metagraph (:obj: bt.metagraph.Metagraph):
             Previous state of metagraph before updated resync
         metagraph (:obj: bt.metagraph.Metagraph):
             Latest state of the metagraph with updated uids and hotkeys
    """
    uids_hotkeys_state_dict = dict(zip(previous_metagraph.uids.tolist(), previous_metagraph.hotkeys))
    latest_uids_hotkeys_state_dict = dict(zip(metagraph.uids.tolist(), metagraph.hotkeys))

    updated_uids_indices = []
    for uid, latest_hotkey in latest_uids_hotkeys_state_dict.items():
        if uids_hotkeys_state_dict.get(uid) != latest_hotkey:
            updated_uids_indices.append(uid)

    for index in updated_uids_indices:
        # Reinitialize the bias of the selected index of the linear layer
        torch.nn.init.zeros_(linear_layer.bias[index])
        # Clone the weights of the selected index of the linear layer
        weights = linear_layer.weight[index].clone()
        # Adds a dimension to the weights tensor to make it compatible with the xavier_uniform_ function
        torch.nn.init.xavier_uniform_(weights.unsqueeze(0))
        reinitialized_weights = weights.squeeze(0)
        # Copy the reinitialized weights back to the selected index of the linear layer
        linear_layer.weight[index].data.copy_(reinitialized_weights)


def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def save_state(self):
    r"""Save hotkeys, gating model, neuron model and moving average scores to filesystem."""
    bt.logging.info("save_state()")
    try:
        neuron_state_dict = {
            "neuron_weights": self.moving_averaged_scores,
            "neuron_hotkeys": self.hotkeys,
        }
        torch.save(neuron_state_dict, f"{self.config.neuron.full_path}/model.torch")
        bt.logging.success(
            prefix="Saved model",
            sufix=f"<blue>{ self.config.neuron.full_path }/model.torch</blue>",
        )

        # Save the gating model.
        gating_model_linear_layer_dict = self.gating_model.linear.state_dict()
        gating_model_name = self.config.gating.model_name.replace("/", "_")
        gating_model_file_path = f"{self.config.neuron.full_path}/{gating_model_name}_gating_linear_layer.pth"
        torch.save(gating_model_linear_layer_dict, gating_model_file_path)

        if not self.config.wandb.off and self.config.wandb.track_gating_model:
            model_artifact = wandb.Artifact(f"{gating_model_name}_gating_linear_layer", type="model")
            model_artifact.add_file(gating_model_file_path)
            self.wandb.log_artifact(model_artifact)

        bt.logging.success(prefix="Saved gating model", sufix=f"<blue>{gating_model_file_path}</blue>")
    except Exception as e:
        bt.logging.warning(f"Failed to save model with error: {e}")


def load_state(self):
    r"""Load hotkeys and moving average scores from filesystem."""
    bt.logging.info("load_state()")
    try:
        state_dict = torch.load(f"{self.config.neuron.full_path}/model.torch")
        # Check for nans in saved state dict
        if not torch.isnan(state_dict["neuron_weights"]).any():        
            self.moving_averaged_scores = state_dict["neuron_weights"].clone().detach()
        self.hotkeys = state_dict["neuron_hotkeys"]
        bt.logging.success(
            prefix="Reloaded model",
            sufix=f"<blue>{ self.config.neuron.full_path }/model.torch</blue>",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to load model with error: {e}")
