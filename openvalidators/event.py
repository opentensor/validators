import bittensor as bt
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EventSchema:
    completions: List[str]  # List of completions received for a given prompt
    name: str  # Prompt type, e.g. 'followup', 'answer'
    block: float  # Current block at given step
    gating_loss: float  # Gating model loss for given step
    uids: List[int]  # Queried uids
    prompt: str  # Prompt text string
    step_length: float  # Elapsed time between the beginning of a run step to the end of a run step
    best: str  # Best completion for given prompt

    # Reward data
    rewards: List[float]  # Reward vector for given step
    dahoas_reward_model: Optional[List[float]]
    blacklist_filter: Optional[List[float]]  # Output vector of the blacklist filter
    nsfw_filter: Optional[List[float]]  # Output vector of the nsfw filter
    reciprocate_reward_model: Optional[List[float]]  # Output vector of the reciprocate reward model
    diversity_reward_model: Optional[List[float]]  # Output vector of the diversity reward model
    rlhf_reward_model: Optional[List[float]]  # Output vector of the rlhf reward model
    prompt_reward_model: Optional[List[float]]  # Output vector of the prompt reward model
    relevance_scoring: Optional[List[float]]  # Output vector of the relevance scoring reward model

    # Weights data
    set_weights: Optional[List[List[float]]]

    @staticmethod
    def from_dict(event_dict: dict, should_log_rewards: bool) -> 'EventSchema':
        """Converts a dictionary to an EventSchema object."""
        rewards = {
            'dahoas_reward_model': event_dict.get('dahoas_reward_model'),
            'blacklist_filter': event_dict.get('blacklist_filter'),
            'nsfw_filter': event_dict.get('nsfw_filter'),
            'reciprocate_reward_model': event_dict.get('reciprocate_reward_model'),
            'diversity_reward_model': event_dict.get('diversity_reward_model'),
            'rlhf_reward_model': event_dict.get('rlhf_reward_model'),
            'prompt_reward_model': event_dict.get('prompt_reward_model'),
            'relevance_scoring': event_dict.get('relevance_scoring'),
        }

        # Logs warning that expected data was not set properly
        if should_log_rewards and any(value is None for value in rewards.values()):
            for key, value in rewards.items():
                if value is None:
                    bt.logging.warning(f'EventSchema.from_dict: {key} is None, data will not be logged')

        return EventSchema(
            completions=event_dict['completions'],
            name=event_dict['name'],
            block=event_dict['block'],
            gating_loss=event_dict['gating_loss'],
            uids=event_dict['uids'],
            prompt=event_dict['prompt'],
            step_length=event_dict['step_length'],
            best=event_dict['best'],
            rewards=event_dict['rewards'],
            **rewards,
            set_weights=None,
        )
