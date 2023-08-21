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
import json
import pandas as pd
import openvalidators
import os
from analysis.utils import get_runs, download_data
from traceback import print_exception
from typing import List
from data_formatter import create_json_dataset, create_csv_dataset, create_openai_dataset


DEFAULT_PROJECT = 'opentensor-dev/openvalidators'
DEFAULT_FILTERS = {"tags": {"$in": [openvalidators.__version__]}}


def read_file_into_array(file_path: str) -> List[str]:
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


def create_mining_dataset(
    df: pd.DataFrame,
    export_path: str,
    mining_dataset_output_format: str,
    blacklist_phrases: List[str],
    with_score: bool =False,
    export_openai_dataset: bool = False):
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

    if export_openai_dataset:
        jsonl_dataset = create_openai_dataset(df=df, blacklist=blacklist_phrases)

        with open("openai_mining_dataset_openvalidators.jsonl", "w") as file:
            file.write(jsonl_dataset)

    elif mining_dataset_output_format == 'json':
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
        parser.add_argument("--export_openai_dataset", action="store_true", help="Exports the openai dataset", default=False)

        args = parser.parse_args()

        download_all = args.download_all
        wandb_run_id = args.wandb_run_id
        include_tags = args.include_tags
        export_mining_dataset = args.export_mining_dataset
        export_mining_with_scoring_dataset = args.export_mining_with_scoring_dataset
        export_path = args.export_path
        mining_dataset_output_format = args.mining_dataset_output_format
        export_openai_dataset = args.export_openai_dataset

        bt.logging.info("Current version of openvalidators: " + openvalidators.__version__)

        # Loads the blacklist phrases into an array
        blacklist_phrases = read_file_into_array(args.blacklist_path)

        # Collects dataframe from wandb run logs
        collected_data = collect_data(download_all, export_path, wandb_run_id, include_tags)

        # Creates mining dataset
        if export_mining_dataset or export_mining_with_scoring_dataset or export_openai_dataset:
            create_mining_dataset(
                df=collected_data,
                export_path=export_path,
                mining_dataset_output_format=mining_dataset_output_format,
                blacklist_phrases=blacklist_phrases,
                with_score=export_mining_with_scoring_dataset,
                export_openai_dataset=export_openai_dataset
            )
    except Exception as e:
        bt.logging.error("Error in training loop", str(e))
        bt.logging.debug(print_exception(value=e))
