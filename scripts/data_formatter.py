import pandas as pd
import tqdm
import json
from typing import List
from dataclasses import dataclass

@dataclass
class OpenAISample:
    prompt: str
    completion: str

def create_json_dataset(
    df: pd.DataFrame,
    include_scoring: bool,
    blacklist: List[str]
) -> str:
    dict_dataset = {}

    for _, row in tqdm.tqdm(df.iterrows(), desc='Creating mining dataset', total=len(df), unit='run'):
        base_prompt = row['base_prompt']
        best_followup = row['best_followup']

        answer_prompt = row['answer_prompt']
        best_answer = row['best_answer']

        if best_answer not in blacklist:
            if include_scoring:
                scores = 0
                if isinstance(row["answer_rewards"], list):
                    scores = max(row["answer_rewards"])
                elif isinstance(row["answer_rewards"], float):
                    scores = row["answer_rewards"]

                dict_dataset[answer_prompt] = {best_answer: scores}
            else:
                dict_dataset[answer_prompt] = best_answer

        if best_followup not in blacklist:
            if include_scoring:
                scores = 0
                if isinstance(row["answer_rewards"], list):
                    scores = max(row["answer_rewards"])
                elif isinstance(row["answer_rewards"], float):
                    scores = row["answer_rewards"]
                dict_dataset[base_prompt] = {best_followup: scores}
            else:
                dict_dataset[base_prompt] = best_followup

    return dict_dataset


def create_csv_dataset(
    df: pd.DataFrame,
    include_scoring: bool,
    blacklist: List[str]
) -> pd.DataFrame:
    if include_scoring:
        mining_df = df[['base_prompt', 'best_followup', 'followup_rewards', 'answer_prompt', 'best_answer', 'answer_rewards']]
        # Excludes blacklisted phrases from the dataset
        filtered_df = mining_df[~df['best_followup'].isin(blacklist)]
        filtered_df = filtered_df[~df['best_answer'].isin(blacklist)]

        # Gets the max score for each answer and followup
        filtered_df['followup_rewards'] = filtered_df['followup_rewards'].apply(lambda rewards: max(rewards))
        filtered_df['answer_rewards'] = filtered_df['answer_rewards'].apply(lambda rewards: max(rewards))

        return filtered_df
    else:
        mining_df = df[['base_prompt', 'best_followup', 'answer_prompt', 'best_answer']]
        # Excludes blacklisted phrases from the dataset
        filtered_df = mining_df[~df['best_followup'].isin(blacklist)]
        filtered_df = filtered_df[~df['best_answer'].isin(blacklist)]

        return filtered_df


def create_openai_dataset(
    df: pd.DataFrame,
    blacklist: List[str]
) -> str:
    samples = []

    for _, row in tqdm.tqdm(df.iterrows(), desc='Creating openai mining dataset', total=len(df), unit='run'):
        base_prompt = row['base_prompt']
        best_followup = row['best_followup']

        answer_prompt = row['answer_prompt']
        best_answer = row['best_answer']

        if best_followup not in blacklist:
            samples += [OpenAISample(base_prompt, best_followup)]

        if best_answer not in blacklist:
            samples += [OpenAISample(answer_prompt, best_answer)]

    # Convert dataclass objects to dictionaries
    jsonl_data = "\n".join(
        json.dumps({"prompt": sample.prompt, "completion": sample.completion})
        for sample in samples
    )

    return jsonl_data