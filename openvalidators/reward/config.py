import yaml
import bittensor as bt
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


class RewardFrameworkConfig:
    def __init__(self, config_file_path: str):
        self.rlhf_model_weight = 0
        self.reciprocate_model_weight = 0
        self.dahoas_model_weight = 0
        self.diversity_model_weight = 0
        self.prompt_model_weight = 0
        self.config_file_path = config_file_path

    def load_from_config_file(self):
        with open(self.config_file_path, 'r') as config_file:
            try:
                data = yaml.safe_load(config_file)

                reward_model_weights = data['reward_model_weights']

                self.rlhf_model_weight = reward_model_weights.get(RewardModelType.rlhf.value, 0)
                self.reciprocate_model_weight = reward_model_weights.get(RewardModelType.reciprocate.value, 0)
                self.dahoas_model_weight = reward_model_weights.get(RewardModelType.dahoas.value, 0)
                self.diversity_model_weight = reward_model_weights.get(RewardModelType.diversity.value, 0)
                self.prompt_model_weight = reward_model_weights.get(RewardModelType.prompt.value, 0)

                reward_framework_weights = [self.rlhf_model_weight, self.reciprocate_model_weight,
                                            self.dahoas_model_weight, self.diversity_model_weight,
                                            self.prompt_model_weight]

                if sum(reward_framework_weights) != 1:
                    raise Exception("Sum of reward model weights must be 1")

            except yaml.YAMLError as e:
                bt.logging.error(f"Error while parsing YAML file: {e}")
                raise e