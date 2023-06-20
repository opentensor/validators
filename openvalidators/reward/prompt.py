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

import time
import torch
import bittensor as bt
from .reward import BaseRewardModel
from openvalidators.prompts import FollowupPrompt, AnswerPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM


class PromptRewardModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str: return "prompt-reward-model"

    def __init__(self, device: str):
        super().__init__()
        self.device = device

        # https://huggingface.co/VMware/open-llama-7b-open-instruct
        # Fast tokenizer results in incorrect encoding, set the use_fast = False parameter.
        self.tokenizer = AutoTokenizer.from_pretrained(PromptRewardModel.reward_model_name, use_fast=False)
        # Generative default expects most recent token on right-hand side with padding on left.
        # https://github.com/huggingface/transformers/pull/10552
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(PromptRewardModel.reward_model_name,
                                                          torch_dtype=torch.float16).to(self.device)

    def reward(self, prompt: str, completion: str, name: str) -> float:
        with torch.no_grad():
            # Choose correct scoring prompt for request type.
            if name == 'followup':
                scoring_prompt = FollowupPrompt()
            elif name == 'answer':
                scoring_prompt = AnswerPrompt()

            # Format scoring prompt for this completion.
            scoring_prompt_text = scoring_prompt.text(prompt, completion)

            # Tokenize formatted scoring prompt.
            encodings_dict = self.tokenizer(
                scoring_prompt_text,
                truncation=False,
                max_length=2048,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(self.device)

            # Prompt local reward model.
            start_time = time.time()
            generated_tokens = self.model.generate(input_ids, max_new_tokens=50, max_time=3)
            duration = time.time() - start_time
            generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Extract score from generated text.
            score_text = generated_text[0][len(scoring_prompt_text):]
            score = scoring_prompt.extract_score(score_text)

            # Scale 0-10 score to 0-1 range.
            score /= 10.

            print(f"PromptRewardModel (prompt): {prompt}")
            print(f"PromptRewardModel (completion): {completion}")
            print(f"PromptRewardModel (name): {name}")
            # bt.logging.debug(f"PromptRewardModel (scoring prompt text): {scoring_prompt_text}")
            print(f"Response: {score_text}: {score} [{duration}]")

            return score
