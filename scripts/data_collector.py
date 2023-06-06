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

import bittensor as bt
import argparse
import tqdm
import json
import pandas as pd
import openvalidators
import os
from analysis.utils import get_runs, download_data
from traceback import print_exc


DEFAULT_PROJECT = 'openvalidators'
DEFAULT_FILTERS = {"tags": {"$in": [openvalidators.__version__]}}


def read_file_into_array(file_path: str) -> list[str]:
    """Reads a file into an array of strings"""
    bt.logging.info(f"Loading blacklists phrases from {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def collect_data(
    download_all: bool,
    export_path: str,
    wandb_run_id: str = None,
    include_flags: str = None
) -> pd.DataFrame:
    """Collects data from wandb run logs and exports it to a csv file.
    Args:
        download_all (bool): Whether to download all the data or just a specific run.
        export_path (str): Path to export the data.
        wandb_run_id (str, optional): Wandb run id to download. Defaults to None.
    Returns:
        pd.DataFrame: Dataframe of wandb run logs.
    """
    if download_all:
        if include_flags is not None:
            flags = include_flags.split(',')
            DEFAULT_FILTERS['tags']['$in'].extend(flags)

        bt.logging.info(f'Downloading all files with tags: {DEFAULT_FILTERS}')
        runs = get_runs(DEFAULT_PROJECT, filters=DEFAULT_FILTERS, return_paths=True)
        df = download_data(runs)
        df.to_csv(export_path)
    else:
        if wandb_run_id is None:
            raise Exception("Please specify a wandb run id to download")
        else:
            bt.logging.info(f'Downloading data from run id: {wandb_run_id}')
            df = download_data(f'{DEFAULT_PROJECT}/{wandb_run_id}')
            df.to_csv(export_path)

    bt.logging.info(f'Data collected successfully at: {export_path}')
    return df


def create_json_dataset(
    df: pd.DataFrame,
    include_scoring: bool,
    blacklist: list[str]
) -> dict:
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
    blacklist: list[str]
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


def create_mining_dataset(
    df: pd.DataFrame,
    export_path: str,
    mining_dataset_output_format: str,
    blacklist_phrases: list[str],
    with_score: bool =False):
    """Creates a dataset for mining from the dataframe of wandb run logs.
    Args:
        df (pd.DataFrame): Dataframe of wandb run logs.
        export_path (str): Path to export the dataset.
        with_score (bool, optional): Whether to include the score in the dataset. Defaults to False.
    """
    filename, file_extension = os.path.splitext(export_path)
    mining_dataset_path = f'mining_dataset_{filename}.{mining_dataset_output_format}'

    if with_score:
        mining_dataset_path = f'scored_{mining_dataset_path}'

    bt.logging.info(f"Creating mining dataset: {mining_dataset_path}")

    if mining_dataset_output_format == 'json':
        dict_dataset = create_json_dataset(
            df=df,
            include_scoring=with_score,
            blacklist=blacklist_phrases,
        )
        with open(mining_dataset_path, 'w') as json_file:
            json.dump(dict_dataset, json_file)

    elif mining_dataset_output_format == 'csv':
        df_dataset = create_csv_dataset(
            df=df,
            include_scoring=with_score,
            blacklist=blacklist_phrases,
        )
        df_dataset.to_csv(mining_dataset_path)
    else:
        raise Exception(f"Invalid mining dataset output format: {mining_dataset_output_format}")

    bt.logging.info(f"Mining dataset exported successfully to {mining_dataset_path}")


if __name__ == '__main__':
    try:
        # Create an ArgumentParser object
        parser = argparse.ArgumentParser()

        # Add the flags as parameters
        parser.add_argument("--download_all", action="store_true", help="Downloads all runs from project", default=False)
        parser.add_argument("--wandb_run_id", type=str, help="Specify the wandb run id to download", default=None)
        parser.add_argument("--include_tags", type=str, help="Specify the flags to filter the dataset", default=None)
        parser.add_argument("--export_mining_dataset", action="store_true", help="Exports the mining dataset", default=False)
        parser.add_argument("--export_mining_with_scoring_dataset", action="store_true", help="Exports mining dataset with scores", default=False)
        parser.add_argument("--mining_dataset_output_format", type=str, help="Specify the output format of the mining dataset", default="json")
        parser.add_argument("--export_path", type=str, help="Specify the path to export the dataset", default="validator_dataset.csv")
        parser.add_argument("--blacklist_path", type=str, help="Specify the path to the blacklist phrases", default="blacklist_phrases.txt")

        args = parser.parse_args()

        download_all = args.download_all
        wandb_run_id = args.wandb_run_id
        include_tags = args.include_tags
        export_mining_dataset = args.export_mining_dataset
        export_mining_with_scoring_dataset = args.export_mining_with_scoring_dataset
        export_path = args.export_path
        mining_dataset_output_format = args.mining_dataset_output_format

        bt.logging.info("Current version of openvalidators: " + openvalidators.__version__)

        # Loads the blacklist phrases into an array
        blacklist_phrases = read_file_into_array(args.blacklist_path)

        # Collects dataframe from wandb run logs
        collected_data = collect_data(download_all, export_path, wandb_run_id, include_tags)

        # Creates mining dataset
        if export_mining_dataset or export_mining_with_scoring_dataset:
            create_mining_dataset(
                df=collected_data,
                export_path=export_path,
                mining_dataset_output_format=mining_dataset_output_format,
                blacklist_phrases=blacklist_phrases,
                with_score=export_mining_with_scoring_dataset
            )
    except Exception as e:
        bt.logging.error("Error in training loop", str(e))
        bt.logging.debug(print_exc(e))
